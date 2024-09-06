#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/grammar-vs-llms

rsync -azP data/ euler:/cluster/work/sachan/vilem/grammar-vs-llms/data/
