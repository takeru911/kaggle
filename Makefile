INPUT_DATA_S3_BUCKET=kaggle-input-data

download-input-data: fetch-from-s3

create-project:
ifdef PROJECT
	cp -r template $(PROJECT)
else
	@echo running project is not defined.
	exit 1
endif

fetch-from-s3:
ifdef PROJECT
	aws s3 cp s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input $(PROJECT)/input/ --recursive
else
	@echo running project is not defined.
	exit 1
endif

upload-input-data:
	aws s3 cp  $(PROJECT)/input/ s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input/ --recursive
	@echo $(PROJECT)\'s input data uploaded to s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input/

compress-input-data:
ifdef PROJECT
	tar -zcvf $(PROJECT)/input.tar.gz $(PROJECT)/input
else
	@echo input data is not exist.
	exit 1
endif

init-competition: kaggle $(COMPE_NAME)
	kaggle competitions download -c $(COMPE_NAME) -p $(COMPE_NAME)/input

$(COMPE_NAME):
	mkdir -p $(COMPE_NAME)
	mkdir -p $(COMPE_NAME)/input

kaggle:
	pip install kaggle --user
