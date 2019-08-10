#!/bin/bash

# Pre-process autoMPG.data ready for learning
cat autoMpg.arff | grep -v "[@%?]" | awk  'NF > 0' | cut -d',' -f1-2,4,6- | sed 's/,/ /g' > transformedData.data;

# Split into train, test, and validation sets (60:20:20)
m=$(wc -l transformedData.data | awk '{print $1}')

testLines=$(echo "($m / 100) * 20" | bc)
validationLines=$testLines
trainLines=$(echo "$m - $validationLines - $testLines" | bc)

cat transformedData.data | head -n$trainLines > data/trainData.data
cat transformedData.data | tail -n$validationLines > data/validationData.data
cat transformedData.data | tail -n$(echo "$validationLines + $trainLines" | bc) | head -n$testLines > data/testData.data


