from .hmr import HandModelRecovery


class NetworksFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'generator_base':
            from .generator import Generator
            network = Generator(*args, **kwargs)

        elif network_name == 'generator_spade':
            from .generator import Generator
            network = Generator(*args, **kwargs, spade_layers=[1, 1, 0, 0])

        elif network_name == 'generator_spade_attn':
            from .generator import Generator
            network = Generator(*args, **kwargs, spade_layers=[1, 1, 0, 0], attn_layers=[1,2,3,4,5,6,7,8,9])

        elif network_name == 'generator_spade_attn_tiny':
            from .generator import Generator
            network = Generator(*args, **kwargs, spade_layers=[0, 0, 1, 1], attn_layers=[1,2,3,4,5,6,7,8,9])

        elif network_name == 'discriminator_patch_gan':
            from .discriminator import PatchDiscriminator
            network = PatchDiscriminator(*args, **kwargs)

        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network
