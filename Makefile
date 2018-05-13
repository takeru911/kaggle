INPUT_DATA_S3_BUCKET=kaggle-input-data

download-input-data: fetch-from-s3
ifeq ($(FORCE), true)
	rm -r $(PROJECT)/input
endif
	tar -xvf $(PROJECT)/input.tar.gz
	rm $(PROJECT)/input.tar.gz
	@echo $(PROJECT)\'s input data was download.

fetch-from-s3:
ifdef PROJECT
	aws s3 cp s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input/input.tar.gz $(PROJECT)/
else
	@echo running project is not defined.
	exit 1
endif

upload-input-data: compress-input-data
	aws s3 cp  $(PROJECT)/input.tar.gz s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input/
	rm $(PROJECT)/input.tar.gz
	@echo $(PROJECT)\'s input data uploaded to s3://$(INPUT_DATA_S3_BUCKET)/$(PROJECT)/input/input.tar.gz

compress-input-data:
ifdef PROJECT
	tar -zcvf $(PROJECT)/input.tar.gz $(PROJECT)/input
else
	@echo input data is not exist.
	exit 1
endif
