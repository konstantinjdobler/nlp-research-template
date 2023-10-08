import math


def get_lr_with_cosine_schedule(it, learning_rate, warmup_period, lr_decay_period, min_lr):
    """
    Returns actual lr based on current iteration. Should be called every iteration.
    From lit-gpt: https://github.com/Lightning-AI/lit-gpt/blob/a21d46ae80f84c350ad871578d0348b470c83021/pretrain/redpajama.py#L301
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_period:
        return learning_rate * it / warmup_period
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_period:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_period) / (lr_decay_period - warmup_period)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
