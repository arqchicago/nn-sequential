	?????B???????B??!?????B??	????%B@????%B@!????%B@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????B????D????A???߾??Y???????*	?????)u@2F
Iterator::Model=,Ԛ???!)????V@)	?c???1??D??U@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??q????!??%?@)c?ZB>???15?vE?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ǘ????!??m;#@)S?!?uq{?17Y?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY?8??m??!?& KP?'@)	?^)?p?1?`5k?_??:Preprocessing2U
Iterator::Model::ParallelMapV2?????g?!e??-/h??)?????g?1e??-/h??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice-C??6Z?!?w??=??)-C??6Z?1?w??=??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???V?!???4?v??)Ǻ???V?1???4?v??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??e?c]??!6X??\ @)-C??6J?1?w??=??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 36.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s4.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????%B@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??D??????D????!??D????      ??!       "      ??!       *      ??!       2	???߾?????߾??!???߾??:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????JCPU_ONLYY????%B@b 