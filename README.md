# AIOLOS: Instructions to run all the AIOLOS data processing pipelines in a container

Set the working directory in this folder called ‘aiolos’
Make sure you have DOCKER DESKTOP open.

## Create the docker image
```powershell
docker build . --tag=aiolos_docker_image
```

## Create docker container and run all pipelines with command
If you want your output to go to MongoDB platform you should run the following (ENV=dev):
```powershell
docker run --name aiolos_docker_container --rm --env ENV=dev --env MongoDBUsername="QuintenPipelineUser" --env MongoDBPassword="PDb6uqwogle8sYFL" --env CGM_PASSWORD="cmxJwnSmzPnQa55" aiolos_docker_image
```

If you want your output to go to Impact Healtcare SFTP server you should run the following (ENV=communication_dashboard):
```powershell
docker run --name aiolos_docker_container --rm --env ENV=communication_dashboard --env CGM_PASSWORD="cmxJwnSmzPnQa55" --env CD_PASSWORD="9He:dX@^M747ey" aiolos_docker_image
```