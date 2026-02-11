import torch 
from torchvision import transforms

import pandas as pd
import datetime 
import requests

from io import StringIO
from sklearn.model_selection import train_test_split


def query_weather_db(query, url, token, timeout=180):
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'X-API-Key': token
    }

    try:
        api_response = requests.post(
            url=url, 
            data=query, 
            headers=headers,
            timeout=timeout
        )
        res = pd.read_csv(StringIO(api_response.text), sep=",")
        # api_response.close()
    
    except requests.exceptions.Timeout:
        print('Connection: Query timed out.')
        res = None
        
    finally:
        return res


def get_weather_data(sensor_id, timestamp, weather_bands, freq='8H', eps=1e-5, **kwargs):
    weather_data = query_weather_db(
        f'''"
        SELECT 
            a.ObjectId,
            b.DatastreamId,
            e.PropertyName,
            b.StreamName,
            b.StreamPropertyUnit,
            b.StreamObservationType,
            c.ObservedTimestamp,
            c.ObservedValue
        FROM Objects a
        JOIN Datastreams b on b.ObjectId = a.ObjectId
        JOIN (
            SELECT 
                ObservationId, 
                DatastreamId, 
                LocationId, 
                ObservedTimestamp,
                ObservedValue
            FROM Observations
            WHERE ObservedTimestamp >= unixepoch(\'{str(timestamp)}\') * 1000 AND ObservedTimestamp < unixepoch(\'{str(timestamp + datetime.timedelta(days=1))}\') * 1000
        ) c ON c.DatastreamId = b.DatastreamId
        JOIN Properties e on e.PropertyId = b.PropertyId
        WHERE a.ObjectId = {sensor_id}
        "''', 
        **kwargs
    )

    weather_data = weather_data.loc[
        weather_data.StreamName.isin(weather_bands)
    ].copy()

    res = weather_data.set_index(pd.to_datetime(weather_data.ObservedTimestamp, unit='ms')).groupby(['ObjectId', 'StreamName'], group_keys=True).apply(
        lambda l: getattr(l.resample(freq).ObservedValue, l.StreamObservationType[0].lower())() + eps
    ).sort_index(ascending=True)

    if res.empty:
        return None
    
    res.columns = [col.hour for col in res.columns]
    return res

    
class ITC_Patches(torch.utils.data.Dataset):
    def __init__(self, feats, labels, transform=None, augmentation=None):
        self.input, self.target = torch.cat(feats, dim=0), torch.cat(labels, dim=0) 
        self.pixel_mask = self.target > 0
        
        query = torch.count_nonzero(self.target > 0, dim=(1, 2, 3)) > 0
        self.input, self.target, self.pixel_mask = self.input[query], self.target[query], self.pixel_mask[query]

        self.augmentation = augmentation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Normalize(
                torch.mean(
                    torch.masked.masked_tensor(self.input, self.pixel_mask.repeat(1, self.input.shape[1], 1, 1)),
                    dim=(0, 2, 3)
                ).to_tensor(0),
                torch.var(
                    torch.masked.masked_tensor(self.input, self.pixel_mask.repeat(1, self.input.shape[1], 1, 1)),
                    dim=(0, 2, 3)
                ).to_tensor(0) + 1e-9, # Add a small constant to avoid division by zero, especially in FL setting
            ),
        ])

    def __getitem__(self, item):
        samples, labels, pixel_mask = self.input[item], self.target[item], self.pixel_mask[item]

        if self.augmentation is not None:
            samples_labels_mask = torch.cat([samples, labels, pixel_mask], dim=0)
            samples_labels_mask = self.augmentation(samples_labels_mask)

            samples, labels, pixel_mask = samples_labels_mask[:-2, :, :], samples_labels_mask[[-2], :, :], samples_labels_mask[[-1], :, :] 

        if self.transform is not None:
            # samples, labels = self.transforms(samples, labels)
            samples = self.transform(samples)

        # return samples.astype('float32'), labels.astype('int64')
        return (samples * pixel_mask).float(), labels.float()

    def __len__(self):
        return len(self.target)
