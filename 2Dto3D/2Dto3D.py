import os
import cv2
import json
import torch
import pickle
import numpy as np
import configparser
import scipy.signal
import scipy.ndimage
import torch.nn as nn
from glob import glob
from tqdm import tqdm

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)
    return d, Z, T, b, c

def interpolate(inp, fi): # ref: https://stackoverflow.com/questions/44238581/interpolate-list-to-specific-length
    i, f = int(fi // 1), fi % 1  # Split floating-point index into whole & fractional parts.
    j = i+1 if f > 0 else i  # Avoid index error.
    try:
        return (1-f) * inp[i] + f * inp[j]
    except IndexError:
        return inp[i]

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation. Some dimensions might also be missing
    
    Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns
    orig_data: the input normalized_data, but unnormalized
    """
    T = normalized_data.shape[0] # Batch size
    D = data_mean.shape[0] # Dimensionality
    
    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                    if dim not in dimensions_to_ignore])
    
    orig_data[:, dimensions_to_use] = normalized_data
    
    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data

class ResidualBlock(nn.Module):
    """
    A residual block.
    """
    def __init__(self, linear_size, p_dropout=0.5, kaiming=False, leaky=False):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        
        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCModel(nn.Module):
    def __init__(self,
                 stage_id=1,
                 linear_size=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=False,
                 refine_3d=False, 
                 leaky=False,
                 dm=False,
                 input_size=32,
                 output_size=64):
        """
        Fully-connected network.
        """
        super(FCModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.stage_id = stage_id
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm 
        self.input_size = input_size
        if self.stage_id>1 and self.refine_3d:
            self.input_size += 16 * 3        
        # 3d joints
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.res_blocks = []
        for l in range(num_blocks):
            self.res_blocks.append(ResidualBlock(self.linear_size, 
                                                 self.p_dropout,
                                                 leaky=self.leaky))
        self.res_blocks = nn.ModuleList(self.res_blocks)
        
        # output
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.get_representation(x)
        y = self.w2(y)
        return y
    
    def get_representation(self, x):
        # get the latent representation of an input vector
        # first layer
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # residual blocks
        for i in range(self.num_blocks):
            y = self.res_blocks[i](y)        
        
        return y

if __name__ == '__main__':
    
    SAME_FRAME = 96
    template = np.load('template.npy')
    
    cf = configparser.ConfigParser()
    cf.read("./config.ini")
    model_path = cf.get('PATH', 'MODEL_FOLDER_PATH')
    input_path = cf.get('PATH', 'INPUT_FOLDER_PATH')
    output_path_2d = cf.get('PATH', 'OUTPUT2D_FOLDER_PATH')
    output_path_3d = cf.get('PATH', 'OUTPUT3D_FOLDER_PATH')
    
    # load model
    stats = np.load(os.path.join(model_path, 'stats.npy'), allow_pickle=True).item()
    data_mean_2d = stats['mean_2d']
    dim_to_use_2d = stats['dim_use_2d']
    data_std_2d = stats['std_2d']
    cascade = nn.ModuleList([
        FCModel(stage_id=1, 
                refine_3d=False, 
                norm_twoD=False,
                num_blocks=2, 
                input_size=len(stats['dim_use_2d']), 
                output_size=len(stats['dim_use_3d']), 
                linear_size=1024,
                p_dropout=0.5, 
                leaky=False
                ),
        FCModel(stage_id=2, 
                refine_3d=False, 
                norm_twoD=False,
                num_blocks=2, 
                input_size=len(stats['dim_use_2d']), 
                output_size=len(stats['dim_use_3d']), 
                linear_size=1024,
                p_dropout=0.5, 
                leaky=False
               )
    ])
    cascade.load_state_dict(torch.load(os.path.join(model_path, 'model_dict.pt')))
    cascade.cuda()
    for stage_model in cascade:
        stage_model.eval()
    
    for js in tqdm(glob(os.path.join(input_path, '*.json'))):
        file_name, ext = os.path.splitext(os.path.basename(js))
        pred_list = np.empty([0, 1, 96])
        
        if os.path.getsize(js) == 0: 
            print()
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(file_name + ext, 'is empty')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print()
            continue
            
        with open(js, 'rb') as f:
            sk2D = json.load(f)

        # 2D to 3D       
        for s in sk2D:
            if 'pose_keypoints_2d' not in s['candidates'][0]:
                continue
            s = np.array(s['candidates'][0]['pose_keypoints_2d']).reshape(17, 3)[:, :-1]
            s = s[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        
            skeleton_2d = s.reshape(1, 26)
            skeleton_2d = (skeleton_2d - data_mean_2d[dim_to_use_2d])/data_std_2d[dim_to_use_2d]
            data = torch.from_numpy(skeleton_2d.astype(np.float32))
            data = data.cuda()
            pred = cascade[0](data)
            pred = cascade[1](data) + pred
            pred = unNormalizeData(pred.data.cpu().numpy(),
                                   stats['mean_3d'],
                                   stats['std_3d'],
                                   stats['dim_ignore_3d']
                                   )
            pred = pred.reshape(-1, 96)
            pred_list = np.concatenate((pred_list, np.expand_dims(pred, axis=0)), axis=0)

        # apply median-filter and mean-filter
        kernel = 11
        pred_list = pred_list.reshape(-1, 32, 3)
        for joint in range(pred_list.shape[1]):
            for dim in range(pred_list.shape[2]):
                arr = pred_list[:, joint, dim]
                arr = scipy.signal.medfilt(arr, kernel)
                arr = scipy.ndimage.filters.uniform_filter1d(arr, kernel)
                pred_list[:, joint, dim] = arr

        # fix total frame
        sf_pred_list = np.empty([SAME_FRAME, pred_list.shape[1], pred_list.shape[2]])
        for joint in range(pred_list.shape[1]):
            for xy in range(pred_list.shape[2]):
                inp = list(pred_list[:, joint, xy])
                new_len = SAME_FRAME
                delta = (len(inp)-1) / (new_len - 1)
                outp = [interpolate(inp, i*delta) for i in range(new_len)]
                sf_pred_list[:, joint, xy] = outp
        pred_list = sf_pred_list
        pred_list = pred_list.reshape(-1, 1, 96)

        # transformation on template
        p = np.empty([0, 3])
        for frame in range(SAME_FRAME):
            p = np.concatenate(( p, np.expand_dims( pred_list[frame].reshape(32, 3)[25], axis=0 ) ), axis=0)
            p = np.concatenate(( p, np.expand_dims( pred_list[frame].reshape(32, 3)[17], axis=0 ) ), axis=0)
            p = np.concatenate(( p, np.expand_dims( pred_list[frame].reshape(32, 3)[1], axis=0 ) ), axis=0)
            p = np.concatenate(( p, np.expand_dims( pred_list[frame].reshape(32, 3)[6], axis=0 ) ), axis=0)
        _, Z, T, b, c = compute_similarity_transform(template, p, compute_optimal_scale=True)
        for frame in range(SAME_FRAME):
            pred = pred_list[frame].reshape(-1, 3)
            pred = (b * pred.dot(T)) + c
            pred_list[frame] = pred.reshape(1, 96)

        prjt_2d_list = np.empty([0, 13, 2])
        iter_num = len(pred_list)
        for i in range(iter_num):
            _sk = np.reshape(pred_list[i], (32, -1))[:,0::2]
            prjt_2d = np.array([_sk[1], _sk[2], _sk[3], _sk[6], _sk[7], _sk[8], 
                                  _sk[14], _sk[17], _sk[18], _sk[19], _sk[25], _sk[26], _sk[27]])
            prjt_2d_list = np.concatenate((prjt_2d_list, np.expand_dims(prjt_2d, axis=0)), axis=0)
            
        # save file
        np.save(os.path.join(output_path_2d, file_name + '.npy'), prjt_2d_list)
        pred_list = pred_list.reshape(96, 32, 3)[:, [1, 2, 3, 6, 7, 8, 14, 17, 18, 19, 25, 26, 27]]
        np.save(os.path.join(output_path_3d, file_name + '.npy'), pred_list)
        del prjt_2d_list
        del pred_list
    print('done')