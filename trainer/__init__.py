from .default import get_optimizer, get_scheduler


def get_trainer(name: str):
    if name == 'sft':
        from .default import OurTrainer as Trainer
    else:  # Optionally add more subclasses to handle diff training loops
        from .default import OurTrainer as Trainer
    return Trainer
        