import yaml
import os
import datetime
import getpass
import itertools
from argparse import ArgumentParser

class Config:
    def __parse_args(self):
        parser = ArgumentParser()
        parser.add_argument('parameters', metavar='param', type=str, nargs='*',
                            help='parameters')
        parser.add_argument('--config', dest='name', type=str, help='config name')
        parser.add_argument('--email', dest='no_email', action='store_false', default=True)
        parser.add_argument('--gpu', dest='no_gpu', action='store_false', default=True)
        ns = parser.parse_args()
        self.__runtime_name = ns.name
        self.__parameters = ns.parameters
        self.__no_email = ns.no_email
        self.__no_gpu = ns.no_gpu
    
    def __init__(self):
        self.__parse_args()
        with open('config/static.yaml') as f_config_static:
            with open('config/runtime/{runtime_name}.yaml'.format(runtime_name = self.__runtime_name)) as f_config_runtime:
                self.config_static = yaml.load(f_config_static, Loader=yaml.FullLoader)
                self.config_runtime = yaml.load(f_config_runtime, Loader=yaml.FullLoader)
                self.config_runtime['date'] = datetime.date.today()
                self.config_runtime['config'] = self.__runtime_name

                if self.__no_email:
                    self.config_static['sbatch'] = self.config_static['sbatch'].replace('#SBATCH --mail-type=ALL\n#SBATCH --mail-user={user}@uw.edu\n','')
                
                if self.__no_gpu:
                    self.config_static['sbatch'] = self.config_static['sbatch'].replace('#SBATCH --gres=gpu:1\n','')

                if 'chdir' not in self.config_runtime:
                    self.config_runtime['chdir'] = os.path.dirname(os.path.realpath(__file__))

                if 'user' not in self.config_runtime:
                    self.config_runtime['user'] = getpass.getuser()

    def generate_sbatch_scripts(self):
        parameters = self.config_runtime['parameters']
        for el in itertools.product(* parameters.values()):
            cr = self.config_runtime.copy() # shallow copy
            cr1 = self.config_runtime.copy() # shallow copy
            i = 0
            for k in parameters.keys():
                cr[k] = el[i]
                if type(el[i]) == str:
                    cr1[k] = el[i].replace(' ','_').replace('/','_')
                else:
                    cr1[k] = el[i]
                i = i + 1
            cr['jobname'] = cr['jobname'].format(** cr1)
            yield cr['jobname'], '\n'.join([
                self.config_static.get('sbatch').format(** cr),
                self.config_runtime.get('command').format(** cr)])
    def __remove_email(self):
        pass

    def get(self, name):
        return self.config_runtime[name]
    
def test():
    config = Config()
    
    for s in config.generate_sbatch_scripts():
        pass

    print(config.get('datadir'))
            
if __name__ == "__main__":
    test()
