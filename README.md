# Data Downloader for Media Cloud

A simple but efficient framework to download millions of full-length news articles from well-known media outlets (BBC, NYT, etc.). High-quality language and well-organized paragraphs, which could serve as great corpus for studies on political polarization, propaganda detection, and style understanding. 

We leverage the APIs provided by [Media Cloud](https://mediacloud.org/).

## Getting Started

First replace the api keys with yours in the `api_keys.txt` file ([Register here](https://tools.mediacloud.org/#/user/signup) if you do not have an account).

Then run the `run_download.py` file with your configurations on the following three important fields:

````python
# (in the run_download.py file)
# SET YOUR QUERY TOPICS HERE !!!
query_topics = ["abortion", "gay marriage", "death penalty", ... ]

# SET YOUR PERIOD HERE !!!
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 12, 31)

# (in the media_list.py file)
# SET THE MEDIAS OF YOUR INTERESTS HERE !!!
class Media(Enum):
    # Liberal
    BBC = 1094
    CNN = 1095
    ...

    # Neutral
    CNBC = 1755
    USA = 4
    ...

    # Conservative
    FOX = 1092
    BRB = 19334
    ...
```` 

The script will download full-length news articles from the medias of your interests, within the time period, on the topics pre-defined.
The articles are stored in format of `(year)/(topic)/(hash).json` for storage convenience.

Finally, we can process the json files and convert them into csv files for each year and each topic. We consider an output csv file with the following
attributes:

```text
'title': the title
'author': the author
'media': the name of the source media
'pubdate': the publish date
'words_pos': the POS tagged tokenized sentences, as list of ('word', 'its POS')
'words': the plain text, organized as list of paragraphs
```

Run `run_processing.py` with the proper year selection in the main function. Some basic lexical statistics will also show up.

Note: You may need to install tagger from nltk for the POS.

```python
import nltk
nltk.download('averaged_perceptron_tagger')
```

And that's it already. We will have many topic-named csv files in each year csv output folder, such as `csv_output_2020/marijuana.csv`.

## Performance

Every call to the Media Cloud API is executed through multi-thread. Running the code with my default settings (52 topics, 2020 the whole year, 17 medias) on a machine with Intel(R) Xeon(R) Gold 5218 (2.30GHz, 32 cores, 2 threads / core) will take about 4 hours.  

## Citing

I used this downloader to download the data for my papers published in AAAI 21 and CSCW 21. They are:

[Mitigating Political Bias in Language Models through Reinforced Calibration (AAAI 21 Best Paper Award)](https://ojs.aaai.org/index.php/AAAI/article/view/17744):
```bibtex 
@article{Liu_Jia_Wei_Xu_Wang_Vosoughi_2021,
  title={Mitigating Political Bias in Language Models through Reinforced Calibration},
  volume={35},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/17744},
  number={17},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Liu, Ruibo and Jia, Chenyan and Wei, Jason and Xu, Guangxuan and Wang, Lili and Vosoughi, Soroush},
  year={2021},
  month={May},
  pages={14857-14866}
}
```

[A Transformer-Based Framework for Neutralizing and Reversing the Political Polarity of News Articles](https://www.cs.dartmouth.edu/~rbliu/cscw21.pdf):
```bibtex 
@article{10.1145/3449139,
author = {Liu, Ruibo and Jia, Chenyan and Vosoughi, Soroush},
title = {A Transformer-Based Framework for Neutralizing and Reversing the Political Polarity of News Articles},
year = {2021},
issue_date = {April 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {CSCW1},
url = {https://doi.org/10.1145/3449139},
doi = {10.1145/3449139},
journal = {Proc. ACM Hum.-Comput. Interact.},
month = apr,
articleno = {65},
numpages = {26},
keywords = {political polarization, automatic polarity transfer, selective exposure, journalism, echo chamber, neural networks, transformer-based models}
}
```

Due to copyright concerns I may not publish the exact news articles I downloaded with this repository to the public. You can [contact me](mailto:ruibo.liu.gr@dartmouth.edu) for further questions.

This repository is still under construction. I would upload more interesting scripts later. If you find this repository helpful, feel free to cite my papers!

Enjoy!

