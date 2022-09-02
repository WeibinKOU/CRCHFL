class Scheduler():
    def __init__(self, total_data_size, model_size, entry_size):
        '''
        units:
            total_data_size: GBytes
            model_size: MBytes
            data_entry_size: KBytes

        PS: if total_data_size == -1, means unlimited communication resource
        '''
        self.is_unlimited = True if total_data_size < 0 else False
        self.remain_size = total_data_size * 1024 * 1024
        self.model_size = model_size * 1024
        self.entry_size = entry_size
        if not self.is_unlimited:
            print("Scheduler Initialized data size: ", self.remain_size)

        self.edge_fed_interval = 1
        self.cloud_fed_interval=1
        self.pretrain_epochs=1
        self.pretrain_batch_cnt=5
        self.epochs_after_pretrain=100
        self.wireline_size=0
        self.wireless_size=0

    def transfer_entries(self, entries_cnt):
        ret = True
        if not self.is_unlimited:
            self.remain_size -= entries_cnt * self.entry_size
            ret = True if self.remain_size >=0 else False
            print("After data transferring, Scheduler remaining data size: ", self.remain_size)

        return ret

    def transfer_model(self):
        ret = True
        if not self.is_unlimited:
            self.remain_size -= self.model_size
            ret = True if self.remain_size >=0 else False
            print("After model transferring, Scheduler remaining data size: ", self.remain_size)

        return ret

    def set_edge_fed_interval(self, interval):
        self.edge_fed_interval = interval

    def set_cloud_fed_interval(self, interval):
        self.cloud_fed_interval = interval

    def set_pretrain_epochs(self, epochs):
        self.pretrain_epochs = epochs

    def set_epochs_after_pretrain(self, epochs):
        self.epochs_after_pretrain = epochs

    def set_pretrain_batch_cnt(self, cnt):
        self.pretrain_batch_cnt = cnt

    def wireline_stat(self, size):
        self.wireline_size += size

    def wireless_stat(self, size):
        self.wireless_size += size
