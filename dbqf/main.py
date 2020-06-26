import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


from dbqf.args import parse_args
from dbqf.models import init_model
from dbqf.logging_utils import MLogger
from dbqf.optim import OptimizerFactory
from dbqf.dataloaders import get_loaders


def cleanup():
    dist.destroy_process_group()


def get_checkpoint_base_path():
    base_path = "/checkpoint/" + os.environ["USER"] + "/" + \
                os.environ["SLURM_JOB_ID"] + "/" + \
                os.environ["SLURM_ARRAY_TASK_ID"]
    return base_path


def train(model, optimizer, train_loader_size, test_loader, opt, logger,
          current_batch_size):
    model.train()
    model.to('cuda')

    optimizer.reset()

    for batch_idx in range(current_batch_size, opt.niters):
        if batch_idx % opt.epoch_iters == 0:
            logger.info('Saving checkpoint....')
            optimizer.epoch += 1
            base_path = get_checkpoint_base_path()
            try:
                os.mkdir(base_path)
            except OSError:
                pass
            temp_path = base_path + "/temp.pt"
            state = {
                'epoch': optimizer.epoch,
                'model': model.state_dict(),
                'niters': batch_idx
            }

            # model.train()
            filename = 'checkpoint.pth.tar'
            torch.save(state, temp_path)
            os.replace(temp_path, base_path+'/'+filename)

        loss = optimizer.step()
        if batch_idx % opt.log_interval == 0:
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    optimizer.epoch, batch_idx, train_loader_size,
                    100. * batch_idx / train_loader_size, loss.item()))
            logger.log_value('loss', loss, batch_idx)
            lr = optimizer.optimizer.param_groups[0]['lr']
            logger.log_value('lr', lr, batch_idx)
            logger.log_value('epochs', optimizer.epoch, batch_idx)
            logger.log_value('niters', optimizer.niters, batch_idx)
            logger.log_value('batch_idx', batch_idx, batch_idx)
        if batch_idx % (opt.log_interval * 10) == 0:
            test(model, 'cuda', test_loader, logger, batch_idx)


def test(model, device, test_loader, logger, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    vacc = 100. * correct / len(test_loader.dataset)

    logger.info(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), vacc))
    logger.log_value('Vloss', test_loss, epoch)
    logger.log_value('Vacc', vacc, epoch)
    model.train()


def run_worker(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    world_size = opt.world_size

    logger = MLogger()
    logger.configure(opt.log_dir + '/' + str(0) + '.log',
                     opt.log_dir + '/tb/' + str(0) + '/')

    # create model and move it to GPU with id rank
    model = init_model(opt)
    # model.cuda(local_rank)
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param.fill_(0)

    train_loader, test_loader, train_test_loader = get_loaders(opt)
    train_len = len(train_loader)
    opt.epoch_iters = train_len
    print(opt.epoch_iters)

    optimizer = OptimizerFactory(model, logger, opt, world_size)
    base_path = get_checkpoint_base_path() + '/' + 'checkpoint.pth.tar'
    current_batch_size = 0
    if os.path.isfile(base_path):
        checkpoint = torch.load(base_path)
        epoch = checkpoint['epoch']
        optimizer.niters = checkpoint['niters']
        model.load_state_dict(checkpoint['model'])
        current_batch_size = optimizer.niters
        optimizer.epoch = epoch

    train(model, optimizer,
          train_len, test_loader, opt, logger, current_batch_size)


if __name__ == "__main__":
    opt = parse_args()
    run_worker(opt)
