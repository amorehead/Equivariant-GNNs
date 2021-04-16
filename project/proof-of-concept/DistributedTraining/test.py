import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit,os
import numpy as np
import horovod.torch as hvd

num_iters = 10
num_batches_per_iter = 10
batch_size = 32
fp16_allreduce = False
num_warmup_batches = 10

# Benchmark settings
use_cuda = torch.cuda.is_available()

world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

hvd.init()
if use_cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(local_rank)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, 'resnet50')()

if use_cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none
# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                  named_parameters=model.named_parameters(),
                                  compression=compression)
# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# Set up fixed fake data
data = torch.randn(batch_size, 3, 224, 224)
target = torch.LongTensor(batch_size).random_() % 1000
if use_cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

def log(s, nl=True):
    if world_rank != 0:
        return
    print(s, end='\n' if nl else '')

log('Batch size: %d' % batch_size)
device = 'GPU' if use_cuda else 'CPU'
log('Number of %ss: %d' % (device, world_size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(num_iters):
    time = timeit.timeit(benchmark_step, number=num_batches_per_iter)
    img_sec = batch_size * num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))

