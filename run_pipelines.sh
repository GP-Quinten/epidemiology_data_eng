#!/bin/bash
echo ${ENV}

cd french-paediatric && kedro run --env=${ENV}
echo '----------------------------------'
echo '####  French paediatric pipeline completed  ####'
echo '----------------------------------'

cd ../hcl_data && kedro run --env=${ENV}
echo '----------------------------------'
echo '####  HCL pipeline completed  ####'
echo '----------------------------------'

cd ../vivalto_data && kedro run --env=${ENV}
echo '----------------------------------'
echo '####  Vivalto pipeline completed  ####'
echo '----------------------------------'

cd ../german_data && kedro run --env=${ENV}
echo '----------------------------------'
echo '####  German pipeline completed  ####'
echo '----------------------------------'