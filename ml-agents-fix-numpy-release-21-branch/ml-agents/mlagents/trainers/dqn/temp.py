from tensorboard import SummaryWriter

# Test TensorBoard logging
writer = SummaryWriter("runs/test_run")
writer.add_scalar("Test/Value", 123, 1)
writer.close()
print("TensorBoard test completed!")
