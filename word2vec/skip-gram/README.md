## Word2Vec with SkipGram

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
    epochs=80
    alpha=0.025


#### Word: good

	Epoch: 01 [('great', '0.96185'), ('received', '0.93861'), ('nice', '0.93610'), ('well', '0.93506')]
	Epoch: 10 [('great', '0.98618'), ('nice', '0.97219'), ('way', '0.96242'), ('excellent', '0.95861')]
	Epoch: 20 [('great', '0.98822'), ('nice', '0.96685'), ('excellent', '0.96333'), ('way', '0.95195')]
	Epoch: 30 [('great', '0.98848'), ('excellent', '0.96655'), ('nice', '0.96493'), ('also', '0.95340')]
	Epoch: 40 [('great', '0.98834'), ('excellent', '0.96812'), ('nice', '0.96430'), ('also', '0.95588')]
	Epoch: 50 [('great', '0.98821'), ('excellent', '0.96892'), ('nice', '0.96410'), ('also', '0.95715')]
	Epoch: 60 [('great', '0.98811'), ('excellent', '0.96937'), ('nice', '0.96404'), ('also', '0.95791')]
	Epoch: 70 [('great', '0.98804'), ('excellent', '0.96965'), ('nice', '0.96405'), ('also', '0.95842')]
	Epoch: 80 [('great', '0.98799'), ('excellent', '0.96984'), ('nice', '0.96408'), ('also', '0.95878')]

#### Word: love

	Epoch: 01 [('really', '0.95760'), ('price', '0.95525'), ('beautiful', '0.95320'), ('band', '0.93455')]
	Epoch: 10 [('great', '0.97305'), ('like', '0.97148'), ('beautiful', '0.96918'), ('really', '0.96875')]
	Epoch: 20 [('great', '0.97267'), ('like', '0.96959'), ('beautiful', '0.96376'), ('really', '0.96252')]
	Epoch: 30 [('great', '0.97236'), ('like', '0.96901'), ('beautiful', '0.96151'), ('really', '0.96070')]
	Epoch: 40 [('great', '0.97231'), ('like', '0.96854'), ('beautiful', '0.96058'), ('really', '0.96007')]
	Epoch: 50 [('great', '0.97228'), ('like', '0.96818'), ('beautiful', '0.96009'), ('really', '0.95978')]
	Epoch: 60 [('great', '0.97226'), ('like', '0.96789'), ('beautiful', '0.95980'), ('really', '0.95963')]
	Epoch: 70 [('great', '0.97226'), ('like', '0.96767'), ('beautiful', '0.95961'), ('really', '0.95953')]
	Epoch: 80 [('great', '0.97226'), ('like', '0.96750'), ('beautiful', '0.95948'), ('really', '0.95946')]

#### Word: size

	Epoch: 01 [('work', '0.91579'), ('design', '0.88549'), ('son', '0.88218'), ('expected', '0.88010')]
	Epoch: 10 [('color', '0.93793'), ('little', '0.92741'), ('small', '0.91567'), ('bulky', '0.87910')]
	Epoch: 20 [('little', '0.93282'), ('color', '0.93079'), ('small', '0.91677'), ('large', '0.88349')]
	Epoch: 30 [('little', '0.93472'), ('color', '0.92755'), ('small', '0.91803'), ('face', '0.88619')]
	Epoch: 40 [('little', '0.93586'), ('color', '0.92617'), ('small', '0.91876'), ('face', '0.88870')]
	Epoch: 50 [('little', '0.93662'), ('color', '0.92549'), ('small', '0.91922'), ('face', '0.89034')]
	Epoch: 60 [('little', '0.93717'), ('color', '0.92513'), ('small', '0.91953'), ('face', '0.89149')]
	Epoch: 70 [('little', '0.93758'), ('color', '0.92492'), ('small', '0.91975'), ('face', '0.89232')]
	Epoch: 80 [('little', '0.93789'), ('color', '0.92479'), ('small', '0.91992'), ('face', '0.89295')]

#### Word: brother

	Epoch: 01  [('womans', '0.87032'), ('fashionable', '0.84803'), ('setup', '0.81173'), ('medium', '0.78914')]
	Epoch: 10  [('birthday', '0.93917'), ('dad', '0.93561'), ('precise', '0.89981'), ('law', '0.86294')]
	Epoch: 20  [('birthday', '0.95612'), ('dad', '0.92810'), ('sister', '0.90354'), ('father', '0.90231')]
	Epoch: 30  [('birthday', '0.95687'), ('father', '0.92647'), ('dad', '0.92029'), ('sister', '0.91059')]
	Epoch: 40  [('birthday', '0.95630'), ('father', '0.93451'), ('dad', '0.91592'), ('sister', '0.91338')]
	Epoch: 50  [('birthday', '0.95568'), ('father', '0.93833'), ('sister', '0.91487'), ('dad', '0.91322')]
	Epoch: 60  [('birthday', '0.95516'), ('father', '0.94052'), ('sister', '0.91580'), ('dad', '0.91141')]
	Epoch: 70  [('birthday', '0.95473'), ('father', '0.94193'), ('sister', '0.91644'), ('nephew', '0.91095')]
	Epoch: 80  [('birthday', '0.95439'), ('father', '0.94290'), ('sister', '0.91691'), ('nephew', '0.91145')]
