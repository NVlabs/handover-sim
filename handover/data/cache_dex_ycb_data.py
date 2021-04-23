from handover.envs.dex_ycb import DexYCB


def main():
  print('Caching DexYCB data')

  dex_ycb = DexYCB()
  dex_ycb.save_to_cache()

  print('Done')


if __name__ == '__main__':
  main()
