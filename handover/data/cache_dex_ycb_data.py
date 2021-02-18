from handover.envs.dex_ycb import DexYCB


def main():
  print('Caching DexYCB data')

  dex_ycb = DexYCB(load_cache=False)
  dex_ycb.save_cache()

  print('Done')


if __name__ == '__main__':
  main()
