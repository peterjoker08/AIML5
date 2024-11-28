from tensorboardX import SummaryWriter

writer = SummaryWriter("runs/test_run")
for i in range(100):
    writer.add_scalar("TestMetric", i, i)
    print(i)
writer.close()
