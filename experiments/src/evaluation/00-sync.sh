#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/grammar-vs-llms

rsync -azP data/translated/ euler:/cluster/work/sachan/vilem/grammar-vs-llms/data/translated/
rsync -azP euler:/cluster/work/sachan/vilem/grammar-vs-llms/data/evaluated/ data/evaluated/
