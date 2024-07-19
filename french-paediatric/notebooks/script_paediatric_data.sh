# Script to run the HCL Data Preprocessing Pipeline for HCL labs.

#!/bin/bash

# Step 1: Clone the repository (if not already done)
# git clone <https://XXXXXX/@bitbucket.org/aiolos-project/hcl_data.git>
# cd hcl_data

# Step 2: Activate the Kedro virtual environment
# Use the appropriate command to activate your virtual environment (e.g., source venv/bin/activate)

# Step 3: Install the kedro-docker plugin
pip install kedro-docker

# Step 4: Build the Docker container
kedro docker build

# Step 5: Set environment variables (replace placeholders with your credentials)
Set-Item -Path Env:AWS_ACCESS_KEY_ID -Value "YOUR_AWS_ACCESS_KEY_ID"
Set-Item -Path Env:AWS_SECRET_ACCESS_KEY -Value "YOUR_AWS_SECRET_ACCESS_KEY"
Set-Item -Path Env:MongoDBUsername -Value "YOUR_MongoDBUsername"
Set-Item -Path Env:MongoDBPassword -Value "YOUR_MongoDBPassword"
Set-Item -Path Env:CGM_PASSWORD -Value "YOUR_CGM_PASSWORD"


# Step 6: Run the entire pipeline with Docker container in 'dev' environment
kedro docker run --pipeline=paediatric_data_processing --env=dev --docker-args="--env AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY --env MongoDBUsername=MongoDBUsername --env MongoDBPassword=MongoDBPassword --env CGM_PASSWORD=CGM_PASSWORD"

# Step 7: Optionally, run the pipeline locally for development (outside Docker)
kedro run --pipeline=paediatric_data_processing --env=dev

# Step 8: Optionally, visualize the pipeline (outside Docker)
kedro viz --pipeline=paediatric_data_processing --env=dev

# Step 9: Deactivate the virtual environment (if it was activated)
# deactivate
