import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
import tensorflow as tf

sentences = [
"Welcome to the inception institute of artificial intelligence, my name is Tony Robbins! I am kidding, I am not really Tony Robbins. and I never said this. I am still a work in progress. so, don't mind any strange artifacts you might hear."
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  #fmin_list=[125,115,105,95,85,75,65,55]
  #fmax_list=[7600,6600,5600,4600,3600]
  
  fmin_list=[15,25,35,45,65,75,85,95,105,115,135,145,155,165,175]
  fmax_list=[7600]
  #print(hparams_debug_string())
  for fmin in fmin_list:
    for fmax in fmax_list:
      hparams.fmin=fmin
      hparams.fmax=fmax
      synth = Synthesizer(reuse=tf.AUTO_REUSE)
      synth.load(args.checkpoint)
      base_path = get_output_base_path(args.checkpoint)
      for i, text in enumerate(sentences):
        path = '%s-fmin_%d-fmax_%d-%03d.wav' % (base_path,fmin,fmax,i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
          f.write(synth.synthesize(text))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
