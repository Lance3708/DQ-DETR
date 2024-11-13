from torch.utils.tensorboard import SummaryWriter

# 初始化 TensorBoard SummaryWriter
if args.rank == 0:  # 仅在主进程中初始化
    writer = SummaryWriter(log_dir=args.output_dir)

# 记录一些示例信息
writer.add_text("Config", json.dumps(vars(args), indent=2))
writer.add_scalar("Training/Loss", loss_value, global_step)
writer.add_scalar("Training/Accuracy", accuracy_value, global_step)

# 在适当的位置关闭 SummaryWriter
if args.rank == 0:
    writer.close()