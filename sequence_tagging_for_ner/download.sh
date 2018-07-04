if [ -f assignment2.zip ]; then
    echo "data exist"
else
    wget http://cs224d.stanford.edu/assignment2/assignment2.zip
fi

if [ $? -eq 0  ];then
    unzip assignment2.zip
    mkdir -p /root/.cache/paddle/dataset/data
    cp assignment2_release/data/ner/wordVectors.txt /root/.cache/paddle/dataset/data
    cp assignment2_release/data/ner/vocab.txt /root/.cache/paddle/dataset/data
    rm -rf assignment2.zip assignment2_release
else
  echo "download data error!" >> /dev/stderr
  exit 1
fi
