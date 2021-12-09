## Word2Vec with CBOW

Applied Softmax activation function at output.

### Data

    dataset: Watch Review from Amazon
    filter: word count from 10 to 30, removed stop word & other than [a-z]
    sentences: 3806
    words: 33643
    vocab: 1327

### Setup

    window_size=2
    embedding_size=8
    epochs=60
    alpha=0.025


#### Word: good

	Epoch: 01 [('loved', '0.93163'), ('great', '0.90588'), ('price', '0.88385'), ('wear', '0.85689')]
	Epoch: 10 [('great', '0.95345'), ('well', '0.91165'), ('classy', '0.90042'), ('nice', '0.89792')]
	Epoch: 20 [('great', '0.95801'), ('classy', '0.91883'), ('nice', '0.90574'), ('well', '0.90195')]
	Epoch: 30 [('great', '0.96017'), ('classy', '0.90954'), ('nice', '0.90731'), ('well', '0.89991')]
	Epoch: 40 [('great', '0.96143'), ('nice', '0.90806'), ('classy', '0.90187'), ('like', '0.89958')]
	Epoch: 50 [('great', '0.96198'), ('nice', '0.90840'), ('like', '0.90062'), ('well', '0.89930')]
	Epoch: 60 [('great', '0.96205'), ('nice', '0.90843'), ('like', '0.90074'), ('well', '0.89930')]

#### Word: love

	Epoch: 01 [('band', '0.94629'), ('christmas', '0.85685'), ('beautiful', '0.84278'), ('perfect', '0.84136')]
	Epoch: 10 [('classy', '0.94703'), ('awesome', '0.94610'), ('like', '0.94372'), ('beautiful', '0.94230')]
	Epoch: 20 [('classy', '0.96104'), ('affordable', '0.95967'), ('like', '0.95618'), ('beautiful', '0.92391')]
	Epoch: 30 [('like', '0.96022'), ('affordable', '0.95724'), ('classy', '0.95468'), ('enjoy', '0.92212')]
	Epoch: 40 [('like', '0.96191'), ('affordable', '0.95179'), ('classy', '0.94947'), ('enjoy', '0.92361')]
	Epoch: 50 [('like', '0.96271'), ('affordable', '0.94773'), ('classy', '0.94615'), ('enjoy', '0.92422')]
	Epoch: 60 [('like', '0.96286'), ('affordable', '0.94686'), ('classy', '0.94546'), ('enjoy', '0.92432')]

#### Word: size

	Epoch: 01 [('local', '0.93030'), ('readable', '0.90371'), ('outfits', '0.88956'), ('world', '0.87308')]
	Epoch: 10 [('little', '0.95955'), ('band', '0.92243'), ('bit', '0.91356'), ('use', '0.89700')]
	Epoch: 20 [('little', '0.94126'), ('band', '0.91803'), ('fit', '0.90114'), ('heavy', '0.88300')]
	Epoch: 30 [('little', '0.93036'), ('band', '0.91163'), ('fit', '0.90785'), ('heavy', '0.88733')]
	Epoch: 40 [('little', '0.92490'), ('fit', '0.90832'), ('band', '0.90813'), ('heavy', '0.88629')]
	Epoch: 50 [('little', '0.92243'), ('fit', '0.90804'), ('band', '0.90649'), ('heavy', '0.88530')]
	Epoch: 60 [('little', '0.92175'), ('fit', '0.90791'), ('band', '0.90603'), ('heavy', '0.88497')]

#### Word: brother

	Epoch: 01  [('womans', '0.89807'), ('fashionable', '0.89060'), ('ever', '0.80464'), ('ordered', '0.79815')]
	Epoch: 10  [('least', '0.85220'), ('jeans', '0.83350'), ('last', '0.82712'), ('wasnt', '0.81699')]
	Epoch: 20  [('titanium', '0.89555'), ('least', '0.87735'), ('last', '0.86747'), ('ive', '0.84832')]
	Epoch: 30  [('given', '0.87034'), ('titanium', '0.86092'), ('least', '0.85139'), ('ive', '0.84881')]
	Epoch: 40  [('given', '0.88426'), ('two', '0.87226'), ('likes', '0.86673'), ('ive', '0.83580')]
	Epoch: 50  [('given', '0.88691'), ('two', '0.88159'), ('likes', '0.86984'), ('ive', '0.82764')]
	Epoch: 60  [('given', '0.88735'), ('two', '0.88419'), ('likes', '0.87024'), ('ive', '0.82434')]

