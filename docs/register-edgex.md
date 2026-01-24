# Register Profile in EdgeX
After install and setup EdgeX to your device, you can register your device service in the EdgeX platform. Detailed instructions can be found in [this link](https://docs.edgexfoundry.org/3.2/walk-through/Ch-WalkthroughDeviceService/).
## Register device profile
Send the POST request to : `your_address:59881/api/v2/deviceprofile`
```
[{
    "requestId": "19d2339b-190a-4b6b-bf7c-510f4858395c",
    "apiVersion": "v2",
    "profile": {
        "name": "DFLNode",
        "manufacturer": "JNU",
        "labels": ["device", "profile"],
        "deviceResources": [{
            "name": "Models",
            "properties": {
                "valueType": "String",
                "readWrite": "RW"
            }
        }],
        "deviceCommands": [{
            "name": "Switch",
            "readWrite": "RW",
            "resourceOperations": [{
                "deviceResource": "Models"
            }]
        }]
    }
}]

```

## Register device service
Send the POST request to : `your_address:59881/api/v2/deviceservice`
```
[{
    "requestId": "e6e8a2f4-eb14-4649-9e2b-175247911369",
    "apiVersion": "v2",
    "service": {
        "name": "deviceservice011",
        "adminState": "UNLOCKED",
        "operatingState": "UP",
        "labels": ["device", "service"],
    }
}]
```

## Register device name
Send the POST request to : `your_address:59881/api/v2/devices`
```
[{
    "apiVersion": "v2",
    "device": {
        "name": "Node",
        "adminState": "UNLOCKED",
        "operatingState": "UP",
        "labels": ["http", "device"],
        "serviceName": "deviceservice011",
        "profileName": "DFLNode",
        "protocols": {
            "http-test": {
                "Address": "192.168.0.55",
                "Port": "8080"
            }
        }
    }
}]

```

## Create strem flow in EdgeX
Send the POST request to : `your_address:59720/streams`
```
curl -X POST \
  http://localhost:59720/streams \
  -H 'Content-Type: application/json' \
  -d '{
  "sql": "create stream demo() WITH (FORMAT=\"JSON\", TYPE=\"edgex\")"
}'

```