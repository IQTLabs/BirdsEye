from sigscan import SigScan


def test_sigscan():
    instance = SigScan(config_path='tests/test_sigscan_config.ini')
    instance.main()
    instance = SigScan(config_path='tests/test_sigscan_config2.ini')
    instance.main()
