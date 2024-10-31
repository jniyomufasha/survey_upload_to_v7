# survey_upload_to_v7

This repository contains code for uploading survey images and, if necessary, their annotations, to V7.

## Using
* Clone the repository
* You may need to activate a python virtual environment
* Install the dependencies
  ```
  pip install -r requirements.txt
  ```
* Open the project in an IDE, like VS code
* In the repository root folder, create a `secrets.json` file and fill in your V7 API key like
  ```
  {"v7_api_key": "Your API key"}
  ```
* In the `src/main.py` file, list all projects you want to process in the `projects_to_process` list on line 38.
* In the terminal in which you will run the script, make sure you paste in the S3 access keys or authenticate whichever way you do it. But you need to be authenticated to access S3.
* Run the `src/main.py` file
```
python src/main.py
```
* Wait for it to run. The more the projects, the longer it will take. This will upload the images with towers if any are found in their respective project folders in V7.

**Note:** This will not upload annotations. When images are pushed to V7, it take a moment to process them. Uploading annotations to an image that isn't processed yet fails. We upload annotations to images that have been processed. The way to upload annotations is described in the *analysis file* section.


## The analysis file
A `src/analysis.ipynb` file has been created to give visibility into the projects that have been analyzed **on our local system** and the stats of their results.

This file will be useful to double check whether the image processign has run successfully. For example, it has a check to see whether all images listed in the project's `images.json` file in S3 are reported to have been processed locally. That will be disblayed as `project_is_fully_processed` column in the `projects_summary_df`.

This file will also give visibility into the projects in which no tower was detected, since these will not appear in V7. Any project with `num_images_with_towers` as 0 in the `project_is_fully_processed` will not have any images detected by the model, hence won't appear in V7.

If, for some unexpected reason, a folder has `num_images_with_towers > 0` yet it is not in V7, although this should not happen, the file gives an `upload_project` function which can be used to upload it.

Since the annotations are not uploaded in the `src/main.py` script as we have to wait for the images to be processed first, this notebook can be used to upload annotations. In the **Uploading annotations for multiple projects** section of the notebook, you list the projects for which you need to upload annotations and run.

**Attention**: Uploading annotations to images that already have annotations will cause duplicated annotations. We didn't want to override anyone's work.
