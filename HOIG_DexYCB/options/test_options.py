from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--output_dir', type=str, default='./outputs/results/',
                                  help='output directory to save the results')
        self._parser.add_argument('--eval_pairs', type=str, default='eval_pairs.pkl',
                                  help='directory to load the evaluation pairs')

        self._parser.add_argument('--bg_model', type=str,
                                  default='ORIGINAL',
                                  help='if it is `ORIGINAL`, it will use the '
                                       'original BGNet of the generator of LiquidWarping GAN, (default)'
                                       'otherwise, use a pretrained background inpaintor.')

        # visualizer
        self._parser.add_argument('--ip', type=str, default='http://localhost', help='visdom ip')
        self._parser.add_argument('--port', type=int, default=8097, help='visdom port')

        # save results or not
        self._parser.add_argument('--save_res', action='store_true', default=False,
                                  help='save images or not, if true, the results are saved in `${output_dir}/preds`.')

        self.is_train = False
