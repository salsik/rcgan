from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class trainingDataset(Dataset):
    def __init__(self, datasetA, datasetB, n_frames=128):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames

        self.length = min(len(dataset_A), len(dataset_B))

        num_samples = min(len(dataset_A), len(dataset_B))
        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))
        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_A = list()
        train_data_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            train_data_A.append(data_A[:, start_A:end_A])

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            train_data_B.append(data_B[:, start_B:end_B])

        train_data_A = np.array(train_data_A)
        train_data_B = np.array(train_data_B)

        return train_data_A[index], train_data_B[index]

    def __len__(self):
        return min(len(self.datasetA), len(self.datasetB))




class trainingRCGANDataset(Dataset):
    def __init__(self, datasets, n_frames=128):
        self.datasets = datasets
        self.n_frames = n_frames

    def __getitem__(self, index):

        #dataset_A = self.datasetA
        #dataset_B = self.datasetB
        n_frames = self.n_frames


        _datasets = self.datasets
        n_frames = self.n_frames



        #self.length = min(len(datasets_), len(dataset_B))

        self.length = min(map(len, _datasets))

        #num_samples = min(len(dataset_A), len(dataset_B))
        num_samples = min(map(len, _datasets))

        #train_data_A_idx = np.arange(len(dataset_A))
        #train_data_B_idx = np.arange(len(dataset_B))

        train_data_idx = list(map(np.arange,map(len, _datasets)))


        #np.random.shuffle(train_data_A_idx)
        #np.random.shuffle(train_data_B_idx)


        [np.random.shuffle(sublist) for sublist in train_data_idx]

        #train_data_A_idx_subset = train_data_A_idx[:num_samples]
        #train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_idx_subset = [sublist[:num_samples] for sublist in train_data_idx]



       # train_data_A = list()
      #  train_data_B = list()

       # train_data_idx_subset = [sublist[:10] for sublist in train_data_idx]
        train_data = list()

        k=0
        for elem in train_data_idx_subset:
            train_data_temp=list()
            for idx_A in zip(elem):
               # print(idx_A)
                data_A = _datasets[k][int(idx_A[0])]
                frames_A_total = data_A.shape[1]
                assert frames_A_total >= n_frames
                start_A = np.random.randint(frames_A_total - n_frames + 1)
                end_A = start_A + n_frames
                train_data_temp.append(data_A[:, start_A:end_A])
            train_data_temp= np.array(train_data_temp)
            train_data.append(train_data_temp)
            k += 1


        #
        # for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        #     data_A = dataset_A[idx_A]
        #     frames_A_total = data_A.shape[1]
        #     assert frames_A_total >= n_frames
        #     start_A = np.random.randint(frames_A_total - n_frames + 1)
        #     end_A = start_A + n_frames
        #     train_data_A.append(data_A[:, start_A:end_A])
        #
        #     data_B = dataset_B[idx_B]
        #     frames_B_total = data_B.shape[1]
        #     assert frames_B_total >= n_frames
        #     start_B = np.random.randint(frames_B_total - n_frames + 1)
        #     end_B = start_B + n_frames
        #     train_data_B.append(data_B[:, start_B:end_B])

       # train_data_A = np.array(train_data_A)
       # train_data_B = np.array(train_data_B)

        return [item[index] for item in train_data]
       # return train_data_A[index], train_data_B[index]

    def __len__(self):
        return   min(map(len, self.datasets))


def testAssertion():
    a=53
    b=70
    try:
        assert b>a,"assertionxcvcxvx error"
        print("b is larger")
    except Exception as e:
        print("exception in assertion is", str(e)) #raise  AssertionError
    else:
        print("1234444")

if __name__ == '__main__':
    trainA = np.random.randn(200, 44, 1554)
    trainB = np.random.randn(1580, 24, 1554)
    trainC = np.random.randn(120, 64, 1554)
    trainD = np.random.randn(180, 54, 1554)
    #dataset = trainingDataset(trainA, trainB)

    testAssertion()
    datasets = []

    datasets.append(trainA)
    datasets.append(trainB)
    datasets.append(trainC)
    datasets.append(trainD)

    dataset = trainingRCGANDataset(datasets,n_frames=40)

   # print(len(dataset.__getitem__(100)))
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)
    for epoch in range(1):
        for i, elem in enumerate(trainLoader):
            print(i)
            for key in elem:
                print(key.shape)
