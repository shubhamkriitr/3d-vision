# Test dataloader by running this snippet
# Will output input data of first chosen instance-pair

""" run test """

from visn.data.loader import GroupedImagesDataset, SequentialDataLoader

dataset = GroupedImagesDataset()
dataloader = SequentialDataLoader(dataset=dataset, config={"batch_size": 1})

iter = dataloader.__iter__()
next = dataloader.__next__()

print(next)
