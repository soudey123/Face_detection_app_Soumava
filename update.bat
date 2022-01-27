az acr build --registry CtnRegistry1  --resource-group WebApps --image faceapp .  && \
az webapp create -g WebApps -p AppServicePlan -n faceapp -i ctnregistry1.azurecr.io/faceapp:latest