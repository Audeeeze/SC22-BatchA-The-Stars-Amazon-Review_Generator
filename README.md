# Amazon Review Generator 
The amazing team `@Kai Lorenc` `@Srishti Deb` `@Jonathan Setiabudi` `@Anatulya Nandi` `@Ariel Gonzalez` trained the GPT-2 model and designed the webapp using html/css & flask.
## Overview
TODO: 
1. short discription: who are the users, what problems are we trying to solve
2. what is the models/technology 
## App Preview 
![home](https://github.com/Siqi-Fang/SC22-BatchA-The-Stars-Amazon-Review_Generator/blob/master/home.png)
![result](https://github.com/Siqi-Fang/SC22-BatchA-The-Stars-Amazon-Review_Generator/blob/master/result.png)

## Dataset 
TODO:
3. what's the dataset 

## Extension 
TODO:
4.what are some extension features

## How to run 
1. On your terminal, navigate to a directory where you want this app to be stored. (recommended: create a virtual environmnet using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html))

2. `git clone` this repository and cd into `app/` directory.

3. Download dependencies with `python3 -m pip install -r requirements.txt`. (if this command fail to download all the dependencies, use `pip3 install <packagename>` to download missing ones.)

4. Move the model files into 'app/model' folder. See `model/drive_link.txt` for our trained model. (You should have 2 folders, `pos_model` and `neg_model`, inside `model/`)

5. Change line 99 of `app/main.py`, the value of website_url to `'localhost'`.

6. Run app with `python3 -m main`. You should see something like `http://127.0.0.1/12345`, open it in your browser.

