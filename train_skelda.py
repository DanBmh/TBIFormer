import copy
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim as optim
import torch_dct as dct  # https://github.com/zh217/torch-dct
import time
import random
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from models.net import TBIFormer
from utils.opt import Options
from utils.soft_dtw_cuda import SoftDTW
from utils.dataloader import Data
from utils.metrics import FDE, JPE, APE
from utils.TRPE import bulding_TRPE_matrix

sys.path.append("/PoseForecasters/")
import utils_pipeline

# ==================================================================================================

datamode = "gt-gt"
# datamode = "pred-pred"

sconfig = {
    "item_step": 2,
    "window_step": 2,
    # "item_step": 1,
    # "window_step": 1,
    "select_joints": [
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ],
}

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_rpt.json",
]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"


# ==================================================================================================


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def temporal_partition(src, opt):
    src = src[:,:,1:]
    B, N, L, _ = src.size()
    stride = 1
    fn = int((L - opt.kernel_size) / stride + 1)
    idx = np.expand_dims(np.arange(opt.kernel_size), axis=0) + \
          np.expand_dims(np.arange(fn), axis=1) * stride
    return idx

def train(model, batch_data, opt):
    input_seq, output_seq = batch_data
    B, N, _, D = input_seq.shape
    input_ = input_seq.view(-1, 50, input_seq.shape[-1])
    output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])
    
    trj_dist = bulding_TRPE_matrix(input_seq.reshape(B,N,-1,13,3), opt)  #  trajectory similarity distance

    offset = input_[:, 1:50, :] - input_[:, :49, :]  #   dispacement sequence
    src = dct.dct(offset)

    rec_ = model.forward(src, N,  trj_dist)
    rec = dct.idct(rec_)
    results = output_[:, :1, :]
    for i in range(1, 26):
        results = torch.cat(
            [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
            dim=1)
    results = results[:, 1:, :]  # 3 13 45

    rec_loss = torch.mean((rec[:, :25, :] - (output_[:, 1:26, :] - output_[:, :25, :])) ** 2)



    prediction = results.view(B, N, -1, 13, 3)
    gt = output_.view(B, N, -1, 13, 3)[:,:,1:,...]

    return prediction, gt, rec_loss, results


def process_data(batch):

    sequences_train = utils_pipeline.make_input_sequence(
        batch, "input", datamode, make_relative=False
    )
    sequences_gt = utils_pipeline.make_input_sequence(
        batch, "target", datamode, make_relative=False
    )

    # Add last input frame to the target sequence
    sequences_gt = np.concatenate(
        [sequences_train[:, -1:, :, :], sequences_gt], axis=1
    )

    # Switch y and z axes
    sequences_train = sequences_train[:, :, :, [0, 2, 1]]
    sequences_gt = sequences_gt[:, :, :, [0, 2, 1]]

    # Reshape to [nbatch, npersons, nframes, njoints * 3]
    nbatch = sequences_train.shape[0]
    sequences_train = sequences_train.reshape(
        [nbatch, 1, sequences_train.shape[1], -1]
    )
    sequences_gt = sequences_gt.reshape([nbatch, 1, sequences_gt.shape[1], -1])

    # # Duplicate persons 3 times and add an x offset to each
    # sequences_train = np.repeat(sequences_train, 3, axis=1)
    # sequences_gt = np.repeat(sequences_gt, 3, axis=1)
    # offsets = np.array([0, 2, -2]).reshape(1, 3, 1, 1)
    # sequences_train[:, :, :, 0::3] += offsets
    # sequences_gt[:, :, :, 0::3] += offsets

    return sequences_train, sequences_gt


def processor(opt):

    device = opt.device

    setup_seed(opt.seed)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # dataset = Data(dataset='mocap_umpm', mode=0, device=device, transform=False, opt=opt)
    # test_dataset = Data(dataset='mocap_umpm', mode=1, device=device, transform=False, opt=opt)

    print(stamp)
    # dataloader = DataLoader(dataset,
    #                         batch_size=opt.train_batch,
    #                         shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=opt.test_batch,
    #                              shuffle=False, drop_last=True)

    sconfig["input_n"] = opt.input_time
    sconfig["output_n"] = opt.output_time
    opt.joint_dim = len(sconfig["select_joints"]) * 3

    # Load preprocessed datasets
    print("Loading datasets ...")
    if opt.ckpt_load_path != "":
        dataset_test, dlen_test = utils_pipeline.load_dataset(
            dataset_eval_test.format("test"), "test", sconfig
        )
        dataset_test = dataset_test["sequences"]
    else:
        dataset_train, dlen_train = [], 0
        for dp in datasets_train:
            ds, dlen = utils_pipeline.load_dataset(dp, "train", sconfig)
            dataset_train.extend(ds["sequences"])
            dlen_train += dlen
        dataset_eval, dlen_eval = utils_pipeline.load_dataset(
            dataset_eval_test.format("eval"), "eval", sconfig
        )
        dataset_eval = dataset_eval["sequences"]


    model = TBIFormer(input_dim=opt.d_model, d_model=opt.d_model,
                        d_inner=opt.d_inner, n_layers=opt.num_stage,
                        n_head=opt.n_head , d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, device=device,kernel_size=opt.kernel_size, opt=opt).to(device)



    print(">>> training params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    if opt.ckpt_load_path != "":
        opt.test_batch = 1
        optimizer = None

        checkpoint = torch.load(opt.ckpt_load_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        label_gen_test = utils_pipeline.create_labels_generator(dataset_test, sconfig)

        frame_idx = [5, 10, 15, 20, 25]
        n = 0
        jpe_err_total = np.arange(len(frame_idx), dtype = np.float_)

        with torch.no_grad():

            nbatch = opt.test_batch
            for batch in tqdm.tqdm(
                utils_pipeline.batch_iterate(label_gen_test, batch_size=nbatch),
                total=int(dlen_test / nbatch),
            ):

                # Process data
                sequences_train, sequences_gt = process_data(batch)

                sequences_train = torch.from_numpy(sequences_train).to(device)
                sequences_gt = torch.from_numpy(sequences_gt).to(device)
                batch_data = [sequences_train, sequences_gt]

                prediction, gt, test_loss, _ = train(model, batch_data, opt)
                jpe_err = JPE(gt, prediction, frame_idx)

                jpe_err_total += jpe_err
                n += nbatch

        print("=== JPE Test Error ===")
        print([jpe / n for jpe in jpe_err_total])
        exit()

    Evaluate = True
    save_model = True
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr=opt.lr)


    loss_min = 100
    for epoch_i in range(1, opt.epochs+1):
        with torch.autograd.set_detect_anomaly(True):
            model.train()
        loss_list=[]
        test_loss_list=[]
        """
        ==================================
           Training Processing
        ==================================
        """
        label_gen_train = utils_pipeline.create_labels_generator(dataset_train, sconfig)
        label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, sconfig)

        nbatch = opt.train_batch
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
            total=int(dlen_train / nbatch),
        ):

            # Process data
            sequences_train, sequences_gt = process_data(batch)

            sequences_train = torch.from_numpy(sequences_train).to(device)
            sequences_gt = torch.from_numpy(sequences_gt).to(device)
            batch_data = [sequences_train, sequences_gt]

            # print(sequences_train[0,0])
            # print(sequences_gt[0,0])
            # exit()

            # import vis_skelda
            # vis_skelda.visualize(sequences_train, sequences_gt)
            # exit()

            _, _, loss, _ = train(model, batch_data, opt)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=10000)
            optimizer.step()
            loss_list.append(loss.item())

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }

        loss_cur = np.mean(loss_list)
        print('epoch:', epoch_i, 'loss:', loss_cur, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))
        if save_model:
            # if (epoch_i + 1) % 5 == 0:
            save_path = os.path.join('checkpoints', f'epoch_{epoch_i}.model')
            torch.save(checkpoint, save_path)



        frame_idx = [5, 10, 15, 20, 25]
        n = 0
        ape_err_total = np.arange(len(frame_idx), dtype = np.float_)
        jpe_err_total = np.arange(len(frame_idx), dtype = np.float_)
        fde_err_total = np.arange(len(frame_idx), dtype = np.float_)

        if Evaluate:
            with torch.no_grad():
                """
                  ==================================
                     Validating Processing
                  ==================================
                  """
                model.eval()
                print("\033[0:35mEvaluating.....\033[m")

                nbatch = opt.train_batch
                for batch in tqdm.tqdm(
                    utils_pipeline.batch_iterate(label_gen_eval, batch_size=nbatch),
                    total=int(dlen_eval / nbatch),
                ):

                    # Process data
                    sequences_train, sequences_gt = process_data(batch)

                    sequences_train = torch.from_numpy(sequences_train).to(device)
                    sequences_gt = torch.from_numpy(sequences_gt).to(device)
                    batch_data = [sequences_train, sequences_gt]

                    n += 1
                    prediction, gt, test_loss, _ = train(model, batch_data, opt)
                    test_loss_list.append(test_loss.item())

                    ape_err = APE(gt, prediction, frame_idx)
                    jpe_err = JPE(gt, prediction, frame_idx)
                    fde_err = FDE(gt, prediction, frame_idx)

                    ape_err_total += ape_err
                    jpe_err_total += jpe_err
                    fde_err_total += fde_err

                test_loss_cur = np.mean(test_loss_list)

                if test_loss_cur < loss_min:
                    save_path = os.path.join('checkpoints', "h36m-25fps-1s", f'best_epoch.model')
                    torch.save(checkpoint, save_path)
                    loss_min = test_loss_cur
                    print(f"Best epoch_{checkpoint['epoch']} model is saved!")



                print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}".format("Lengths", 200, 400, 600, 800, 1000))
                print("=== JPE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", jpe_err_total[0]/n,
                                                                                            jpe_err_total[1] / n,
                                                                                            jpe_err_total[2]/n,
                                                                                            jpe_err_total[3]/n,
                                                                                            jpe_err_total[4]/n ))
                print("=== APE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", ape_err_total[0] / n,
                                                                                            ape_err_total[1] / n,
                                                                                            ape_err_total[2] / n,
                                                                                            ape_err_total[3] / n,
                                                                                            ape_err_total[4] / n))
                print("=== FDE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", fde_err_total[0] / n,
                                                                                            fde_err_total[1] / n,
                                                                                            fde_err_total[2] / n,
                                                                                            fde_err_total[3] / n,
                                                                                            fde_err_total[4] / n))

if __name__ == '__main__':
    option = Options().parse()
    processor(option)
