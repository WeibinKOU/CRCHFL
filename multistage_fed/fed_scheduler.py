class Scheduler():
    def __init__(self, total_data_size, model_size, entry_size):
        '''
        units:
            total_data_size: GBytes
            model_size: MBytes
            data_entry_size: KBytes
        '''
        self.remain_size = total_data_size * 1024 * 1024
        self.model_size = model_size * 1024
        self.entry_size = entry_size
        print("Scheduler Initialized data size: ", self.remain_size)

    def transfer_entries(self, entries_cnt):
        self.remain_size -= entries_cnt * self.entry_size
        ret = True if self.remain_size >=0 else False
        self.remain_size = self.remain_size if ret else 0
        print("After data transferring, Scheduler remaining data size: ", self.remain_size)

        return ret

    def transfer_model(self):
        self.remain_size -= self.model_size
        ret = True if self.remain_size >=0 else False
        self.remain_size = self.remain_size if ret else 0
        print("After model transferring, Scheduler remaining data size: ", self.remain_size)

        return ret
