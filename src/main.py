import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from apex import amp  # uncomment lines related with `amp` to use apex

from dataset import MyDataset
from model import DeepDTAF, test


print(sys.argv)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
SHOW_PROCESS_BAR = True
data_path = '../data/'
seed = np.random.randint(33927, 33928) ##random 
path = Path(f'../runs/DeepDTAF_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
device = torch.device("cuda:0")  # or torch.device('cpu')
            
max_seq_len = 1000  
max_pkt_len = 63
max_smi_len = 150

batch_size = 16
n_epoch = 20
interrupt = None
save_best_epoch = 13 #  when `save_best_epoch` is reached and the loss starts to decrease, save best model parameters
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

writer = SummaryWriter(path)
f_param = open(path / 'parameters.txt', 'w')

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')
f_param.write(f'device={device}\n'
          f'seed={seed}\n'
          f'write to {path}\n')
               

print(f'max_seq_len={max_seq_len}\n'
      f'max_pkt_len={max_pkt_len}\n'
      f'max_smi_len={max_smi_len}')

f_param.write(f'max_seq_len={max_seq_len}\n'
      f'max_pkt_len={max_pkt_len}\n'
      f'max_smi_len={max_smi_len}\n')


assert 0<save_best_epoch<n_epoch

model = DeepDTAF()
model = model.to(device)
print(model)
f_param.write('model: \n')
f_param.write(str(model)+'\n')
f_param.close()

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None),
                               batch_size=batch_size,
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True)
                for phase_name in ['training', 'validation', 'test']}
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, epochs=n_epoch,
                                          steps_per_epoch=len(data_loaders['training']))
loss_function = nn.MSELoss(reduction='sum')

# fp16
model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
        
start = datetime.now()
print('start at ', start)

best_epoch = -1
best_val_loss = 100000000
for epoch in range(1, n_epoch + 1):
    tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
    for idx, (*x, y) in tbar:
        model.train()

        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(*x)
        loss = loss_function(output.view(-1), y.view(-1))
            
        # fp16
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         loss.backward() 
            
        optimizer.step()
        scheduler.step()

        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(y):.3f}')

    for _p in ['training', 'validation']:
        performance = test(model, data_loaders[_p], loss_function, device, False)
        for i in performance:
            writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)
        if _p=='validation' and epoch>=save_best_epoch and performance['loss']<best_val_loss:
            best_val_loss = performance['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), path / 'best_model.pt')
            
            
model.load_state_dict(torch.load(path / 'best_model.pt'))
with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in ['training', 'validation', 'test',]:
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
