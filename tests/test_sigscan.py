from sigscan import SigScan


def test_sigscan():
    instance = SigScan(config='tests/test_sigscan_config.ini')
    instance.main()
