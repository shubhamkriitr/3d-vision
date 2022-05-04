from visn.data.loader import GroupedImagesDataset, SequentialDataLoader

dataset = GroupedImagesDataset()
dataloader = SequentialDataLoader(dataset=dataset, batch_size=2)

iter = dataloader.__iter__()
next = dataloader.__next__()

print(next)
