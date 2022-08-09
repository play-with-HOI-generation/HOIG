class ModelsFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None
        if model_name == 'trainer':
            from .trainer import Trainer
            model = Trainer(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model
