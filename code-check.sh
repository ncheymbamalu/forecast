#!/usr/bin/env bash
tools=( black isort mypy flake8 )
for tool in ${tools[*]}
  do
  echo "starting $tool..."
  eval "$tool src"
  sleep 1
  done

files=( 
    src/ingest.py 
    src/transform.py 
    src/train.py 
    src/forecast.py 
    src/evaluate.py 
    src/pipeline.py
    app.py
    )

for file in ${files[*]}
  do
  echo "checking $file..."
  pylint $file
  done
