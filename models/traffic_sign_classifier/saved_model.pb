??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718æ

?
layer_1_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*&
shared_namelayer_1_conv2d/kernel
?
)layer_1_conv2d/kernel/Read/ReadVariableOpReadVariableOplayer_1_conv2d/kernel*&
_output_shapes
:<*
dtype0
~
layer_1_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_namelayer_1_conv2d/bias
w
'layer_1_conv2d/bias/Read/ReadVariableOpReadVariableOplayer_1_conv2d/bias*
_output_shapes
:<*
dtype0
?
layer_2_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*(
shared_namelayer_2_conv2d_2/kernel
?
+layer_2_conv2d_2/kernel/Read/ReadVariableOpReadVariableOplayer_2_conv2d_2/kernel*&
_output_shapes
:<<*
dtype0
?
layer_2_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*&
shared_namelayer_2_conv2d_2/bias
{
)layer_2_conv2d_2/bias/Read/ReadVariableOpReadVariableOplayer_2_conv2d_2/bias*
_output_shapes
:<*
dtype0
?
layer_4_conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_namelayer_4_conv2d_3/kernel
?
+layer_4_conv2d_3/kernel/Read/ReadVariableOpReadVariableOplayer_4_conv2d_3/kernel*&
_output_shapes
:<*
dtype0
?
layer_4_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namelayer_4_conv2d_3/bias
{
)layer_4_conv2d_3/bias/Read/ReadVariableOpReadVariableOplayer_4_conv2d_3/bias*
_output_shapes
:*
dtype0
?
layer_5_conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelayer_5_conv2d_4/kernel
?
+layer_5_conv2d_4/kernel/Read/ReadVariableOpReadVariableOplayer_5_conv2d_4/kernel*&
_output_shapes
:*
dtype0
?
layer_5_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namelayer_5_conv2d_4/bias
{
)layer_5_conv2d_4/bias/Read/ReadVariableOpReadVariableOplayer_5_conv2d_4/bias*
_output_shapes
:*
dtype0
?
layer_8_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_namelayer_8_dense_1/kernel
?
*layer_8_dense_1/kernel/Read/ReadVariableOpReadVariableOplayer_8_dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
layer_8_dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namelayer_8_dense_1/bias
z
(layer_8_dense_1/bias/Read/ReadVariableOpReadVariableOplayer_8_dense_1/bias*
_output_shapes	
:?*
dtype0
?
layer_10_dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*(
shared_namelayer_10_dense_2/kernel
?
+layer_10_dense_2/kernel/Read/ReadVariableOpReadVariableOplayer_10_dense_2/kernel*
_output_shapes
:	?+*
dtype0
?
layer_10_dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*&
shared_namelayer_10_dense_2/bias
{
)layer_10_dense_2/bias/Read/ReadVariableOpReadVariableOplayer_10_dense_2/bias*
_output_shapes
:+*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/layer_1_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_nameAdam/layer_1_conv2d/kernel/m
?
0Adam/layer_1_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_1_conv2d/kernel/m*&
_output_shapes
:<*
dtype0
?
Adam/layer_1_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*+
shared_nameAdam/layer_1_conv2d/bias/m
?
.Adam/layer_1_conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_1_conv2d/bias/m*
_output_shapes
:<*
dtype0
?
Adam/layer_2_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name Adam/layer_2_conv2d_2/kernel/m
?
2Adam/layer_2_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_2_conv2d_2/kernel/m*&
_output_shapes
:<<*
dtype0
?
Adam/layer_2_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_nameAdam/layer_2_conv2d_2/bias/m
?
0Adam/layer_2_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_2_conv2d_2/bias/m*
_output_shapes
:<*
dtype0
?
Adam/layer_4_conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name Adam/layer_4_conv2d_3/kernel/m
?
2Adam/layer_4_conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_4_conv2d_3/kernel/m*&
_output_shapes
:<*
dtype0
?
Adam/layer_4_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/layer_4_conv2d_3/bias/m
?
0Adam/layer_4_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_4_conv2d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer_5_conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/layer_5_conv2d_4/kernel/m
?
2Adam/layer_5_conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_5_conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/layer_5_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/layer_5_conv2d_4/bias/m
?
0Adam/layer_5_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_5_conv2d_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer_8_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/layer_8_dense_1/kernel/m
?
1Adam/layer_8_dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_8_dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/layer_8_dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/layer_8_dense_1/bias/m
?
/Adam/layer_8_dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_8_dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/layer_10_dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*/
shared_name Adam/layer_10_dense_2/kernel/m
?
2Adam/layer_10_dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_10_dense_2/kernel/m*
_output_shapes
:	?+*
dtype0
?
Adam/layer_10_dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*-
shared_nameAdam/layer_10_dense_2/bias/m
?
0Adam/layer_10_dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_10_dense_2/bias/m*
_output_shapes
:+*
dtype0
?
Adam/layer_1_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_nameAdam/layer_1_conv2d/kernel/v
?
0Adam/layer_1_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_1_conv2d/kernel/v*&
_output_shapes
:<*
dtype0
?
Adam/layer_1_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*+
shared_nameAdam/layer_1_conv2d/bias/v
?
.Adam/layer_1_conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_1_conv2d/bias/v*
_output_shapes
:<*
dtype0
?
Adam/layer_2_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name Adam/layer_2_conv2d_2/kernel/v
?
2Adam/layer_2_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_2_conv2d_2/kernel/v*&
_output_shapes
:<<*
dtype0
?
Adam/layer_2_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_nameAdam/layer_2_conv2d_2/bias/v
?
0Adam/layer_2_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_2_conv2d_2/bias/v*
_output_shapes
:<*
dtype0
?
Adam/layer_4_conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name Adam/layer_4_conv2d_3/kernel/v
?
2Adam/layer_4_conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_4_conv2d_3/kernel/v*&
_output_shapes
:<*
dtype0
?
Adam/layer_4_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/layer_4_conv2d_3/bias/v
?
0Adam/layer_4_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_4_conv2d_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer_5_conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/layer_5_conv2d_4/kernel/v
?
2Adam/layer_5_conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_5_conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/layer_5_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/layer_5_conv2d_4/bias/v
?
0Adam/layer_5_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_5_conv2d_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer_8_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/layer_8_dense_1/kernel/v
?
1Adam/layer_8_dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_8_dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/layer_8_dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/layer_8_dense_1/bias/v
?
/Adam/layer_8_dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_8_dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/layer_10_dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?+*/
shared_name Adam/layer_10_dense_2/kernel/v
?
2Adam/layer_10_dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_10_dense_2/kernel/v*
_output_shapes
:	?+*
dtype0
?
Adam/layer_10_dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*-
shared_nameAdam/layer_10_dense_2/bias/v
?
0Adam/layer_10_dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_10_dense_2/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?JB?J B?J
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratem?m?m?m?!m?"m?'m?(m?5m?6m??m?@m?v?v?v?v?!v?"v?'v?(v?5v?6v??v?@v?
V
0
1
2
3
!4
"5
'6
(7
58
69
?10
@11
V
0
1
2
3
!4
"5
'6
(7
58
69
?10
@11
 
?
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
Lmetrics
regularization_losses

Mlayers
Nnon_trainable_variables
 
a_
VARIABLE_VALUElayer_1_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElayer_1_conv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
Olayer_regularization_losses
Player_metrics

Qlayers
trainable_variables
regularization_losses
Rmetrics
Snon_trainable_variables
ca
VARIABLE_VALUElayer_2_conv2d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElayer_2_conv2d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
Tlayer_regularization_losses
Ulayer_metrics

Vlayers
trainable_variables
regularization_losses
Wmetrics
Xnon_trainable_variables
 
 
 
?
	variables
Ylayer_regularization_losses
Zlayer_metrics

[layers
trainable_variables
regularization_losses
\metrics
]non_trainable_variables
ca
VARIABLE_VALUElayer_4_conv2d_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElayer_4_conv2d_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
#	variables
^layer_regularization_losses
_layer_metrics

`layers
$trainable_variables
%regularization_losses
ametrics
bnon_trainable_variables
ca
VARIABLE_VALUElayer_5_conv2d_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElayer_5_conv2d_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)	variables
clayer_regularization_losses
dlayer_metrics

elayers
*trainable_variables
+regularization_losses
fmetrics
gnon_trainable_variables
 
 
 
?
-	variables
hlayer_regularization_losses
ilayer_metrics

jlayers
.trainable_variables
/regularization_losses
kmetrics
lnon_trainable_variables
 
 
 
?
1	variables
mlayer_regularization_losses
nlayer_metrics

olayers
2trainable_variables
3regularization_losses
pmetrics
qnon_trainable_variables
b`
VARIABLE_VALUElayer_8_dense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElayer_8_dense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
7	variables
rlayer_regularization_losses
slayer_metrics

tlayers
8trainable_variables
9regularization_losses
umetrics
vnon_trainable_variables
 
 
 
?
;	variables
wlayer_regularization_losses
xlayer_metrics

ylayers
<trainable_variables
=regularization_losses
zmetrics
{non_trainable_variables
ca
VARIABLE_VALUElayer_10_dense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElayer_10_dense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
A	variables
|layer_regularization_losses
}layer_metrics

~layers
Btrainable_variables
Cregularization_losses
metrics
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/layer_1_conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/layer_1_conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_2_conv2d_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_2_conv2d_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_4_conv2d_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_4_conv2d_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_5_conv2d_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_5_conv2d_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_8_dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/layer_8_dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_10_dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_10_dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_1_conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/layer_1_conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_2_conv2d_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_2_conv2d_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_4_conv2d_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_4_conv2d_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_5_conv2d_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_5_conv2d_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_8_dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/layer_8_dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_10_dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_10_dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
$serving_default_layer_1_conv2d_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCall$serving_default_layer_1_conv2d_inputlayer_1_conv2d/kernellayer_1_conv2d/biaslayer_2_conv2d_2/kernellayer_2_conv2d_2/biaslayer_4_conv2d_3/kernellayer_4_conv2d_3/biaslayer_5_conv2d_4/kernellayer_5_conv2d_4/biaslayer_8_dense_1/kernellayer_8_dense_1/biaslayer_10_dense_2/kernellayer_10_dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_101396
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)layer_1_conv2d/kernel/Read/ReadVariableOp'layer_1_conv2d/bias/Read/ReadVariableOp+layer_2_conv2d_2/kernel/Read/ReadVariableOp)layer_2_conv2d_2/bias/Read/ReadVariableOp+layer_4_conv2d_3/kernel/Read/ReadVariableOp)layer_4_conv2d_3/bias/Read/ReadVariableOp+layer_5_conv2d_4/kernel/Read/ReadVariableOp)layer_5_conv2d_4/bias/Read/ReadVariableOp*layer_8_dense_1/kernel/Read/ReadVariableOp(layer_8_dense_1/bias/Read/ReadVariableOp+layer_10_dense_2/kernel/Read/ReadVariableOp)layer_10_dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/layer_1_conv2d/kernel/m/Read/ReadVariableOp.Adam/layer_1_conv2d/bias/m/Read/ReadVariableOp2Adam/layer_2_conv2d_2/kernel/m/Read/ReadVariableOp0Adam/layer_2_conv2d_2/bias/m/Read/ReadVariableOp2Adam/layer_4_conv2d_3/kernel/m/Read/ReadVariableOp0Adam/layer_4_conv2d_3/bias/m/Read/ReadVariableOp2Adam/layer_5_conv2d_4/kernel/m/Read/ReadVariableOp0Adam/layer_5_conv2d_4/bias/m/Read/ReadVariableOp1Adam/layer_8_dense_1/kernel/m/Read/ReadVariableOp/Adam/layer_8_dense_1/bias/m/Read/ReadVariableOp2Adam/layer_10_dense_2/kernel/m/Read/ReadVariableOp0Adam/layer_10_dense_2/bias/m/Read/ReadVariableOp0Adam/layer_1_conv2d/kernel/v/Read/ReadVariableOp.Adam/layer_1_conv2d/bias/v/Read/ReadVariableOp2Adam/layer_2_conv2d_2/kernel/v/Read/ReadVariableOp0Adam/layer_2_conv2d_2/bias/v/Read/ReadVariableOp2Adam/layer_4_conv2d_3/kernel/v/Read/ReadVariableOp0Adam/layer_4_conv2d_3/bias/v/Read/ReadVariableOp2Adam/layer_5_conv2d_4/kernel/v/Read/ReadVariableOp0Adam/layer_5_conv2d_4/bias/v/Read/ReadVariableOp1Adam/layer_8_dense_1/kernel/v/Read/ReadVariableOp/Adam/layer_8_dense_1/bias/v/Read/ReadVariableOp2Adam/layer_10_dense_2/kernel/v/Read/ReadVariableOp0Adam/layer_10_dense_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_101879
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_1_conv2d/kernellayer_1_conv2d/biaslayer_2_conv2d_2/kernellayer_2_conv2d_2/biaslayer_4_conv2d_3/kernellayer_4_conv2d_3/biaslayer_5_conv2d_4/kernellayer_5_conv2d_4/biaslayer_8_dense_1/kernellayer_8_dense_1/biaslayer_10_dense_2/kernellayer_10_dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/layer_1_conv2d/kernel/mAdam/layer_1_conv2d/bias/mAdam/layer_2_conv2d_2/kernel/mAdam/layer_2_conv2d_2/bias/mAdam/layer_4_conv2d_3/kernel/mAdam/layer_4_conv2d_3/bias/mAdam/layer_5_conv2d_4/kernel/mAdam/layer_5_conv2d_4/bias/mAdam/layer_8_dense_1/kernel/mAdam/layer_8_dense_1/bias/mAdam/layer_10_dense_2/kernel/mAdam/layer_10_dense_2/bias/mAdam/layer_1_conv2d/kernel/vAdam/layer_1_conv2d/bias/vAdam/layer_2_conv2d_2/kernel/vAdam/layer_2_conv2d_2/bias/vAdam/layer_4_conv2d_3/kernel/vAdam/layer_4_conv2d_3/bias/vAdam/layer_5_conv2d_4/kernel/vAdam/layer_5_conv2d_4/bias/vAdam/layer_8_dense_1/kernel/vAdam/layer_8_dense_1/bias/vAdam/layer_10_dense_2/kernel/vAdam/layer_10_dense_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_102024??
?X
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101505

inputsG
-layer_1_conv2d_conv2d_readvariableop_resource:<<
.layer_1_conv2d_biasadd_readvariableop_resource:<I
/layer_2_conv2d_2_conv2d_readvariableop_resource:<<>
0layer_2_conv2d_2_biasadd_readvariableop_resource:<I
/layer_4_conv2d_3_conv2d_readvariableop_resource:<>
0layer_4_conv2d_3_biasadd_readvariableop_resource:I
/layer_5_conv2d_4_conv2d_readvariableop_resource:>
0layer_5_conv2d_4_biasadd_readvariableop_resource:B
.layer_8_dense_1_matmul_readvariableop_resource:
??>
/layer_8_dense_1_biasadd_readvariableop_resource:	?B
/layer_10_dense_2_matmul_readvariableop_resource:	?+>
0layer_10_dense_2_biasadd_readvariableop_resource:+
identity??'layer_10_dense_2/BiasAdd/ReadVariableOp?&layer_10_dense_2/MatMul/ReadVariableOp?%layer_1_conv2d/BiasAdd/ReadVariableOp?$layer_1_conv2d/Conv2D/ReadVariableOp?'layer_2_conv2d_2/BiasAdd/ReadVariableOp?&layer_2_conv2d_2/Conv2D/ReadVariableOp?'layer_4_conv2d_3/BiasAdd/ReadVariableOp?&layer_4_conv2d_3/Conv2D/ReadVariableOp?'layer_5_conv2d_4/BiasAdd/ReadVariableOp?&layer_5_conv2d_4/Conv2D/ReadVariableOp?&layer_8_dense_1/BiasAdd/ReadVariableOp?%layer_8_dense_1/MatMul/ReadVariableOp?
$layer_1_conv2d/Conv2D/ReadVariableOpReadVariableOp-layer_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02&
$layer_1_conv2d/Conv2D/ReadVariableOp?
layer_1_conv2d/Conv2DConv2Dinputs,layer_1_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
layer_1_conv2d/Conv2D?
%layer_1_conv2d/BiasAdd/ReadVariableOpReadVariableOp.layer_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02'
%layer_1_conv2d/BiasAdd/ReadVariableOp?
layer_1_conv2d/BiasAddBiasAddlayer_1_conv2d/Conv2D:output:0-layer_1_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
layer_1_conv2d/BiasAdd?
layer_1_conv2d/ReluRelulayer_1_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
layer_1_conv2d/Relu?
&layer_2_conv2d_2/Conv2D/ReadVariableOpReadVariableOp/layer_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype02(
&layer_2_conv2d_2/Conv2D/ReadVariableOp?
layer_2_conv2d_2/Conv2DConv2D!layer_1_conv2d/Relu:activations:0.layer_2_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
layer_2_conv2d_2/Conv2D?
'layer_2_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0layer_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02)
'layer_2_conv2d_2/BiasAdd/ReadVariableOp?
layer_2_conv2d_2/BiasAddBiasAdd layer_2_conv2d_2/Conv2D:output:0/layer_2_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
layer_2_conv2d_2/BiasAdd?
layer_2_conv2d_2/ReluRelu!layer_2_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
layer_2_conv2d_2/Relu?
layer_3_pooling_layer_1/MaxPoolMaxPool#layer_2_conv2d_2/Relu:activations:0*/
_output_shapes
:?????????<*
ksize
*
paddingVALID*
strides
2!
layer_3_pooling_layer_1/MaxPool?
&layer_4_conv2d_3/Conv2D/ReadVariableOpReadVariableOp/layer_4_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02(
&layer_4_conv2d_3/Conv2D/ReadVariableOp?
layer_4_conv2d_3/Conv2DConv2D(layer_3_pooling_layer_1/MaxPool:output:0.layer_4_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
layer_4_conv2d_3/Conv2D?
'layer_4_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0layer_4_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'layer_4_conv2d_3/BiasAdd/ReadVariableOp?
layer_4_conv2d_3/BiasAddBiasAdd layer_4_conv2d_3/Conv2D:output:0/layer_4_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
layer_4_conv2d_3/BiasAdd?
layer_4_conv2d_3/ReluRelu!layer_4_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
layer_4_conv2d_3/Relu?
&layer_5_conv2d_4/Conv2D/ReadVariableOpReadVariableOp/layer_5_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&layer_5_conv2d_4/Conv2D/ReadVariableOp?
layer_5_conv2d_4/Conv2DConv2D#layer_4_conv2d_3/Relu:activations:0.layer_5_conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
layer_5_conv2d_4/Conv2D?
'layer_5_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0layer_5_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'layer_5_conv2d_4/BiasAdd/ReadVariableOp?
layer_5_conv2d_4/BiasAddBiasAdd layer_5_conv2d_4/Conv2D:output:0/layer_5_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
layer_5_conv2d_4/BiasAdd?
layer_5_conv2d_4/ReluRelu!layer_5_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
layer_5_conv2d_4/Relu?
layer_6_pooling_layer_2/MaxPoolMaxPool#layer_5_conv2d_4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2!
layer_6_pooling_layer_2/MaxPool
layer_7_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
layer_7_flatten/Const?
layer_7_flatten/ReshapeReshape(layer_6_pooling_layer_2/MaxPool:output:0layer_7_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
layer_7_flatten/Reshape?
%layer_8_dense_1/MatMul/ReadVariableOpReadVariableOp.layer_8_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%layer_8_dense_1/MatMul/ReadVariableOp?
layer_8_dense_1/MatMulMatMul layer_7_flatten/Reshape:output:0-layer_8_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/MatMul?
&layer_8_dense_1/BiasAdd/ReadVariableOpReadVariableOp/layer_8_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_8_dense_1/BiasAdd/ReadVariableOp?
layer_8_dense_1/BiasAddBiasAdd layer_8_dense_1/MatMul:product:0.layer_8_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/BiasAdd?
layer_8_dense_1/ReluRelu layer_8_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/Relu?
layer_9_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2!
layer_9_dropout_1/dropout/Const?
layer_9_dropout_1/dropout/MulMul"layer_8_dense_1/Relu:activations:0(layer_9_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
layer_9_dropout_1/dropout/Mul?
layer_9_dropout_1/dropout/ShapeShape"layer_8_dense_1/Relu:activations:0*
T0*
_output_shapes
:2!
layer_9_dropout_1/dropout/Shape?
6layer_9_dropout_1/dropout/random_uniform/RandomUniformRandomUniform(layer_9_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype028
6layer_9_dropout_1/dropout/random_uniform/RandomUniform?
(layer_9_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(layer_9_dropout_1/dropout/GreaterEqual/y?
&layer_9_dropout_1/dropout/GreaterEqualGreaterEqual?layer_9_dropout_1/dropout/random_uniform/RandomUniform:output:01layer_9_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&layer_9_dropout_1/dropout/GreaterEqual?
layer_9_dropout_1/dropout/CastCast*layer_9_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
layer_9_dropout_1/dropout/Cast?
layer_9_dropout_1/dropout/Mul_1Mul!layer_9_dropout_1/dropout/Mul:z:0"layer_9_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2!
layer_9_dropout_1/dropout/Mul_1?
&layer_10_dense_2/MatMul/ReadVariableOpReadVariableOp/layer_10_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02(
&layer_10_dense_2/MatMul/ReadVariableOp?
layer_10_dense_2/MatMulMatMul#layer_9_dropout_1/dropout/Mul_1:z:0.layer_10_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/MatMul?
'layer_10_dense_2/BiasAdd/ReadVariableOpReadVariableOp0layer_10_dense_2_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02)
'layer_10_dense_2/BiasAdd/ReadVariableOp?
layer_10_dense_2/BiasAddBiasAdd!layer_10_dense_2/MatMul:product:0/layer_10_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/BiasAdd?
layer_10_dense_2/SoftmaxSoftmax!layer_10_dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/Softmax?
IdentityIdentity"layer_10_dense_2/Softmax:softmax:0(^layer_10_dense_2/BiasAdd/ReadVariableOp'^layer_10_dense_2/MatMul/ReadVariableOp&^layer_1_conv2d/BiasAdd/ReadVariableOp%^layer_1_conv2d/Conv2D/ReadVariableOp(^layer_2_conv2d_2/BiasAdd/ReadVariableOp'^layer_2_conv2d_2/Conv2D/ReadVariableOp(^layer_4_conv2d_3/BiasAdd/ReadVariableOp'^layer_4_conv2d_3/Conv2D/ReadVariableOp(^layer_5_conv2d_4/BiasAdd/ReadVariableOp'^layer_5_conv2d_4/Conv2D/ReadVariableOp'^layer_8_dense_1/BiasAdd/ReadVariableOp&^layer_8_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2R
'layer_10_dense_2/BiasAdd/ReadVariableOp'layer_10_dense_2/BiasAdd/ReadVariableOp2P
&layer_10_dense_2/MatMul/ReadVariableOp&layer_10_dense_2/MatMul/ReadVariableOp2N
%layer_1_conv2d/BiasAdd/ReadVariableOp%layer_1_conv2d/BiasAdd/ReadVariableOp2L
$layer_1_conv2d/Conv2D/ReadVariableOp$layer_1_conv2d/Conv2D/ReadVariableOp2R
'layer_2_conv2d_2/BiasAdd/ReadVariableOp'layer_2_conv2d_2/BiasAdd/ReadVariableOp2P
&layer_2_conv2d_2/Conv2D/ReadVariableOp&layer_2_conv2d_2/Conv2D/ReadVariableOp2R
'layer_4_conv2d_3/BiasAdd/ReadVariableOp'layer_4_conv2d_3/BiasAdd/ReadVariableOp2P
&layer_4_conv2d_3/Conv2D/ReadVariableOp&layer_4_conv2d_3/Conv2D/ReadVariableOp2R
'layer_5_conv2d_4/BiasAdd/ReadVariableOp'layer_5_conv2d_4/BiasAdd/ReadVariableOp2P
&layer_5_conv2d_4/Conv2D/ReadVariableOp&layer_5_conv2d_4/Conv2D/ReadVariableOp2P
&layer_8_dense_1/BiasAdd/ReadVariableOp&layer_8_dense_1/BiasAdd/ReadVariableOp2N
%layer_8_dense_1/MatMul/ReadVariableOp%layer_8_dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_100998

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?5
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101227

inputs/
layer_1_conv2d_101192:<#
layer_1_conv2d_101194:<1
layer_2_conv2d_2_101197:<<%
layer_2_conv2d_2_101199:<1
layer_4_conv2d_3_101203:<%
layer_4_conv2d_3_101205:1
layer_5_conv2d_4_101208:%
layer_5_conv2d_4_101210:*
layer_8_dense_1_101215:
??%
layer_8_dense_1_101217:	?*
layer_10_dense_2_101221:	?+%
layer_10_dense_2_101223:+
identity??(layer_10_dense_2/StatefulPartitionedCall?&layer_1_conv2d/StatefulPartitionedCall?(layer_2_conv2d_2/StatefulPartitionedCall?(layer_4_conv2d_3/StatefulPartitionedCall?(layer_5_conv2d_4/StatefulPartitionedCall?'layer_8_dense_1/StatefulPartitionedCall?)layer_9_dropout_1/StatefulPartitionedCall?
&layer_1_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_conv2d_101192layer_1_conv2d_101194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_1009332(
&layer_1_conv2d/StatefulPartitionedCall?
(layer_2_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/layer_1_conv2d/StatefulPartitionedCall:output:0layer_2_conv2d_2_101197layer_2_conv2d_2_101199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_1009502*
(layer_2_conv2d_2/StatefulPartitionedCall?
'layer_3_pooling_layer_1/PartitionedCallPartitionedCall1layer_2_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_1008972)
'layer_3_pooling_layer_1/PartitionedCall?
(layer_4_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall0layer_3_pooling_layer_1/PartitionedCall:output:0layer_4_conv2d_3_101203layer_4_conv2d_3_101205*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_1009682*
(layer_4_conv2d_3/StatefulPartitionedCall?
(layer_5_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall1layer_4_conv2d_3/StatefulPartitionedCall:output:0layer_5_conv2d_4_101208layer_5_conv2d_4_101210*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_1009852*
(layer_5_conv2d_4/StatefulPartitionedCall?
'layer_6_pooling_layer_2/PartitionedCallPartitionedCall1layer_5_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_1009092)
'layer_6_pooling_layer_2/PartitionedCall?
layer_7_flatten/PartitionedCallPartitionedCall0layer_6_pooling_layer_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_1009982!
layer_7_flatten/PartitionedCall?
'layer_8_dense_1/StatefulPartitionedCallStatefulPartitionedCall(layer_7_flatten/PartitionedCall:output:0layer_8_dense_1_101215layer_8_dense_1_101217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_1010112)
'layer_8_dense_1/StatefulPartitionedCall?
)layer_9_dropout_1/StatefulPartitionedCallStatefulPartitionedCall0layer_8_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010992+
)layer_9_dropout_1/StatefulPartitionedCall?
(layer_10_dense_2/StatefulPartitionedCallStatefulPartitionedCall2layer_9_dropout_1/StatefulPartitionedCall:output:0layer_10_dense_2_101221layer_10_dense_2_101223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_1010352*
(layer_10_dense_2/StatefulPartitionedCall?
IdentityIdentity1layer_10_dense_2/StatefulPartitionedCall:output:0)^layer_10_dense_2/StatefulPartitionedCall'^layer_1_conv2d/StatefulPartitionedCall)^layer_2_conv2d_2/StatefulPartitionedCall)^layer_4_conv2d_3/StatefulPartitionedCall)^layer_5_conv2d_4/StatefulPartitionedCall(^layer_8_dense_1/StatefulPartitionedCall*^layer_9_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2T
(layer_10_dense_2/StatefulPartitionedCall(layer_10_dense_2/StatefulPartitionedCall2P
&layer_1_conv2d/StatefulPartitionedCall&layer_1_conv2d/StatefulPartitionedCall2T
(layer_2_conv2d_2/StatefulPartitionedCall(layer_2_conv2d_2/StatefulPartitionedCall2T
(layer_4_conv2d_3/StatefulPartitionedCall(layer_4_conv2d_3/StatefulPartitionedCall2T
(layer_5_conv2d_4/StatefulPartitionedCall(layer_5_conv2d_4/StatefulPartitionedCall2R
'layer_8_dense_1/StatefulPartitionedCall'layer_8_dense_1/StatefulPartitionedCall2V
)layer_9_dropout_1/StatefulPartitionedCall)layer_9_dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_100933

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?3
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101042

inputs/
layer_1_conv2d_100934:<#
layer_1_conv2d_100936:<1
layer_2_conv2d_2_100951:<<%
layer_2_conv2d_2_100953:<1
layer_4_conv2d_3_100969:<%
layer_4_conv2d_3_100971:1
layer_5_conv2d_4_100986:%
layer_5_conv2d_4_100988:*
layer_8_dense_1_101012:
??%
layer_8_dense_1_101014:	?*
layer_10_dense_2_101036:	?+%
layer_10_dense_2_101038:+
identity??(layer_10_dense_2/StatefulPartitionedCall?&layer_1_conv2d/StatefulPartitionedCall?(layer_2_conv2d_2/StatefulPartitionedCall?(layer_4_conv2d_3/StatefulPartitionedCall?(layer_5_conv2d_4/StatefulPartitionedCall?'layer_8_dense_1/StatefulPartitionedCall?
&layer_1_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_conv2d_100934layer_1_conv2d_100936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_1009332(
&layer_1_conv2d/StatefulPartitionedCall?
(layer_2_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/layer_1_conv2d/StatefulPartitionedCall:output:0layer_2_conv2d_2_100951layer_2_conv2d_2_100953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_1009502*
(layer_2_conv2d_2/StatefulPartitionedCall?
'layer_3_pooling_layer_1/PartitionedCallPartitionedCall1layer_2_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_1008972)
'layer_3_pooling_layer_1/PartitionedCall?
(layer_4_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall0layer_3_pooling_layer_1/PartitionedCall:output:0layer_4_conv2d_3_100969layer_4_conv2d_3_100971*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_1009682*
(layer_4_conv2d_3/StatefulPartitionedCall?
(layer_5_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall1layer_4_conv2d_3/StatefulPartitionedCall:output:0layer_5_conv2d_4_100986layer_5_conv2d_4_100988*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_1009852*
(layer_5_conv2d_4/StatefulPartitionedCall?
'layer_6_pooling_layer_2/PartitionedCallPartitionedCall1layer_5_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_1009092)
'layer_6_pooling_layer_2/PartitionedCall?
layer_7_flatten/PartitionedCallPartitionedCall0layer_6_pooling_layer_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_1009982!
layer_7_flatten/PartitionedCall?
'layer_8_dense_1/StatefulPartitionedCallStatefulPartitionedCall(layer_7_flatten/PartitionedCall:output:0layer_8_dense_1_101012layer_8_dense_1_101014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_1010112)
'layer_8_dense_1/StatefulPartitionedCall?
!layer_9_dropout_1/PartitionedCallPartitionedCall0layer_8_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010222#
!layer_9_dropout_1/PartitionedCall?
(layer_10_dense_2/StatefulPartitionedCallStatefulPartitionedCall*layer_9_dropout_1/PartitionedCall:output:0layer_10_dense_2_101036layer_10_dense_2_101038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_1010352*
(layer_10_dense_2/StatefulPartitionedCall?
IdentityIdentity1layer_10_dense_2/StatefulPartitionedCall:output:0)^layer_10_dense_2/StatefulPartitionedCall'^layer_1_conv2d/StatefulPartitionedCall)^layer_2_conv2d_2/StatefulPartitionedCall)^layer_4_conv2d_3/StatefulPartitionedCall)^layer_5_conv2d_4/StatefulPartitionedCall(^layer_8_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2T
(layer_10_dense_2/StatefulPartitionedCall(layer_10_dense_2/StatefulPartitionedCall2P
&layer_1_conv2d/StatefulPartitionedCall&layer_1_conv2d/StatefulPartitionedCall2T
(layer_2_conv2d_2/StatefulPartitionedCall(layer_2_conv2d_2/StatefulPartitionedCall2T
(layer_4_conv2d_3/StatefulPartitionedCall(layer_4_conv2d_3/StatefulPartitionedCall2T
(layer_5_conv2d_4/StatefulPartitionedCall(layer_5_conv2d_4/StatefulPartitionedCall2R
'layer_8_dense_1/StatefulPartitionedCall'layer_8_dense_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
k
2__inference_layer_9_dropout_1_layer_call_fn_101701

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101563

inputs!
unknown:<
	unknown_0:<#
	unknown_1:<<
	unknown_2:<#
	unknown_3:<
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?+

unknown_10:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_1012272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
1__inference_layer_5_conv2d_4_layer_call_fn_101643

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_1009852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
l
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101099

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
T
8__inference_layer_3_pooling_layer_1_layer_call_fn_100903

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_1008972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
1__inference_layer_2_conv2d_2_layer_call_fn_101603

inputs!
unknown:<<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_1009502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101534

inputs!
unknown:<
	unknown_0:<#
	unknown_1:<<
	unknown_2:<#
	unknown_3:<
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?+

unknown_10:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_1010422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_100950

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
L
0__inference_layer_7_flatten_layer_call_fn_101654

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_1009982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_101614

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

?
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_101712

inputs1
matmul_readvariableop_resource:	?+-
biasadd_readvariableop_resource:+
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
T
8__inference_layer_6_pooling_layer_2_layer_call_fn_100915

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_1009092
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
1__inference_layer_10_dense_2_layer_call_fn_101721

inputs
unknown:	?+
	unknown_0:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_1010352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?b
?
__inference__traced_save_101879
file_prefix4
0savev2_layer_1_conv2d_kernel_read_readvariableop2
.savev2_layer_1_conv2d_bias_read_readvariableop6
2savev2_layer_2_conv2d_2_kernel_read_readvariableop4
0savev2_layer_2_conv2d_2_bias_read_readvariableop6
2savev2_layer_4_conv2d_3_kernel_read_readvariableop4
0savev2_layer_4_conv2d_3_bias_read_readvariableop6
2savev2_layer_5_conv2d_4_kernel_read_readvariableop4
0savev2_layer_5_conv2d_4_bias_read_readvariableop5
1savev2_layer_8_dense_1_kernel_read_readvariableop3
/savev2_layer_8_dense_1_bias_read_readvariableop6
2savev2_layer_10_dense_2_kernel_read_readvariableop4
0savev2_layer_10_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_layer_1_conv2d_kernel_m_read_readvariableop9
5savev2_adam_layer_1_conv2d_bias_m_read_readvariableop=
9savev2_adam_layer_2_conv2d_2_kernel_m_read_readvariableop;
7savev2_adam_layer_2_conv2d_2_bias_m_read_readvariableop=
9savev2_adam_layer_4_conv2d_3_kernel_m_read_readvariableop;
7savev2_adam_layer_4_conv2d_3_bias_m_read_readvariableop=
9savev2_adam_layer_5_conv2d_4_kernel_m_read_readvariableop;
7savev2_adam_layer_5_conv2d_4_bias_m_read_readvariableop<
8savev2_adam_layer_8_dense_1_kernel_m_read_readvariableop:
6savev2_adam_layer_8_dense_1_bias_m_read_readvariableop=
9savev2_adam_layer_10_dense_2_kernel_m_read_readvariableop;
7savev2_adam_layer_10_dense_2_bias_m_read_readvariableop;
7savev2_adam_layer_1_conv2d_kernel_v_read_readvariableop9
5savev2_adam_layer_1_conv2d_bias_v_read_readvariableop=
9savev2_adam_layer_2_conv2d_2_kernel_v_read_readvariableop;
7savev2_adam_layer_2_conv2d_2_bias_v_read_readvariableop=
9savev2_adam_layer_4_conv2d_3_kernel_v_read_readvariableop;
7savev2_adam_layer_4_conv2d_3_bias_v_read_readvariableop=
9savev2_adam_layer_5_conv2d_4_kernel_v_read_readvariableop;
7savev2_adam_layer_5_conv2d_4_bias_v_read_readvariableop<
8savev2_adam_layer_8_dense_1_kernel_v_read_readvariableop:
6savev2_adam_layer_8_dense_1_bias_v_read_readvariableop=
9savev2_adam_layer_10_dense_2_kernel_v_read_readvariableop;
7savev2_adam_layer_10_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_layer_1_conv2d_kernel_read_readvariableop.savev2_layer_1_conv2d_bias_read_readvariableop2savev2_layer_2_conv2d_2_kernel_read_readvariableop0savev2_layer_2_conv2d_2_bias_read_readvariableop2savev2_layer_4_conv2d_3_kernel_read_readvariableop0savev2_layer_4_conv2d_3_bias_read_readvariableop2savev2_layer_5_conv2d_4_kernel_read_readvariableop0savev2_layer_5_conv2d_4_bias_read_readvariableop1savev2_layer_8_dense_1_kernel_read_readvariableop/savev2_layer_8_dense_1_bias_read_readvariableop2savev2_layer_10_dense_2_kernel_read_readvariableop0savev2_layer_10_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_layer_1_conv2d_kernel_m_read_readvariableop5savev2_adam_layer_1_conv2d_bias_m_read_readvariableop9savev2_adam_layer_2_conv2d_2_kernel_m_read_readvariableop7savev2_adam_layer_2_conv2d_2_bias_m_read_readvariableop9savev2_adam_layer_4_conv2d_3_kernel_m_read_readvariableop7savev2_adam_layer_4_conv2d_3_bias_m_read_readvariableop9savev2_adam_layer_5_conv2d_4_kernel_m_read_readvariableop7savev2_adam_layer_5_conv2d_4_bias_m_read_readvariableop8savev2_adam_layer_8_dense_1_kernel_m_read_readvariableop6savev2_adam_layer_8_dense_1_bias_m_read_readvariableop9savev2_adam_layer_10_dense_2_kernel_m_read_readvariableop7savev2_adam_layer_10_dense_2_bias_m_read_readvariableop7savev2_adam_layer_1_conv2d_kernel_v_read_readvariableop5savev2_adam_layer_1_conv2d_bias_v_read_readvariableop9savev2_adam_layer_2_conv2d_2_kernel_v_read_readvariableop7savev2_adam_layer_2_conv2d_2_bias_v_read_readvariableop9savev2_adam_layer_4_conv2d_3_kernel_v_read_readvariableop7savev2_adam_layer_4_conv2d_3_bias_v_read_readvariableop9savev2_adam_layer_5_conv2d_4_kernel_v_read_readvariableop7savev2_adam_layer_5_conv2d_4_bias_v_read_readvariableop8savev2_adam_layer_8_dense_1_kernel_v_read_readvariableop6savev2_adam_layer_8_dense_1_bias_v_read_readvariableop9savev2_adam_layer_10_dense_2_kernel_v_read_readvariableop7savev2_adam_layer_10_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :<:<:<<:<:<::::
??:?:	?+:+: : : : : : : : : :<:<:<<:<:<::::
??:?:	?+:+:<:<:<<:<:<::::
??:?:	?+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:<: 

_output_shapes
:<:,(
&
_output_shapes
:<<: 

_output_shapes
:<:,(
&
_output_shapes
:<: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?+: 

_output_shapes
:+:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:<: 

_output_shapes
:<:,(
&
_output_shapes
:<<: 

_output_shapes
:<:,(
&
_output_shapes
:<: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?+: !

_output_shapes
:+:,"(
&
_output_shapes
:<: #

_output_shapes
:<:,$(
&
_output_shapes
:<<: %

_output_shapes
:<:,&(
&
_output_shapes
:<: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::&*"
 
_output_shapes
:
??:!+

_output_shapes	
:?:%,!

_output_shapes
:	?+: -

_output_shapes
:+:.

_output_shapes
: 
?
k
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101679

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_102024
file_prefix@
&assignvariableop_layer_1_conv2d_kernel:<4
&assignvariableop_1_layer_1_conv2d_bias:<D
*assignvariableop_2_layer_2_conv2d_2_kernel:<<6
(assignvariableop_3_layer_2_conv2d_2_bias:<D
*assignvariableop_4_layer_4_conv2d_3_kernel:<6
(assignvariableop_5_layer_4_conv2d_3_bias:D
*assignvariableop_6_layer_5_conv2d_4_kernel:6
(assignvariableop_7_layer_5_conv2d_4_bias:=
)assignvariableop_8_layer_8_dense_1_kernel:
??6
'assignvariableop_9_layer_8_dense_1_bias:	?>
+assignvariableop_10_layer_10_dense_2_kernel:	?+7
)assignvariableop_11_layer_10_dense_2_bias:+'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: J
0assignvariableop_21_adam_layer_1_conv2d_kernel_m:<<
.assignvariableop_22_adam_layer_1_conv2d_bias_m:<L
2assignvariableop_23_adam_layer_2_conv2d_2_kernel_m:<<>
0assignvariableop_24_adam_layer_2_conv2d_2_bias_m:<L
2assignvariableop_25_adam_layer_4_conv2d_3_kernel_m:<>
0assignvariableop_26_adam_layer_4_conv2d_3_bias_m:L
2assignvariableop_27_adam_layer_5_conv2d_4_kernel_m:>
0assignvariableop_28_adam_layer_5_conv2d_4_bias_m:E
1assignvariableop_29_adam_layer_8_dense_1_kernel_m:
??>
/assignvariableop_30_adam_layer_8_dense_1_bias_m:	?E
2assignvariableop_31_adam_layer_10_dense_2_kernel_m:	?+>
0assignvariableop_32_adam_layer_10_dense_2_bias_m:+J
0assignvariableop_33_adam_layer_1_conv2d_kernel_v:<<
.assignvariableop_34_adam_layer_1_conv2d_bias_v:<L
2assignvariableop_35_adam_layer_2_conv2d_2_kernel_v:<<>
0assignvariableop_36_adam_layer_2_conv2d_2_bias_v:<L
2assignvariableop_37_adam_layer_4_conv2d_3_kernel_v:<>
0assignvariableop_38_adam_layer_4_conv2d_3_bias_v:L
2assignvariableop_39_adam_layer_5_conv2d_4_kernel_v:>
0assignvariableop_40_adam_layer_5_conv2d_4_bias_v:E
1assignvariableop_41_adam_layer_8_dense_1_kernel_v:
??>
/assignvariableop_42_adam_layer_8_dense_1_bias_v:	?E
2assignvariableop_43_adam_layer_10_dense_2_kernel_v:	?+>
0assignvariableop_44_adam_layer_10_dense_2_bias_v:+
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp&assignvariableop_layer_1_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_layer_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_layer_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_layer_2_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_layer_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_layer_4_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_layer_5_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_layer_5_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_layer_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_layer_8_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_layer_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_layer_10_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_layer_1_conv2d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_layer_1_conv2d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_layer_2_conv2d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_layer_2_conv2d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_layer_4_conv2d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_layer_4_conv2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_layer_5_conv2d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_layer_5_conv2d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_layer_8_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_layer_8_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_layer_10_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_layer_10_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_layer_1_conv2d_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_layer_1_conv2d_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_layer_2_conv2d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_layer_2_conv2d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_layer_4_conv2d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_layer_4_conv2d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_layer_5_conv2d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_layer_5_conv2d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_layer_8_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_layer_8_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_layer_10_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_layer_10_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_100985

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101069
layer_1_conv2d_input!
unknown:<
	unknown_0:<#
	unknown_1:<<
	unknown_2:<#
	unknown_3:<
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?+

unknown_10:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_1010422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?w
?
!__inference__wrapped_model_100891
layer_1_conv2d_inpute
Klenet_traffic_sign_classifier_layer_1_conv2d_conv2d_readvariableop_resource:<Z
Llenet_traffic_sign_classifier_layer_1_conv2d_biasadd_readvariableop_resource:<g
Mlenet_traffic_sign_classifier_layer_2_conv2d_2_conv2d_readvariableop_resource:<<\
Nlenet_traffic_sign_classifier_layer_2_conv2d_2_biasadd_readvariableop_resource:<g
Mlenet_traffic_sign_classifier_layer_4_conv2d_3_conv2d_readvariableop_resource:<\
Nlenet_traffic_sign_classifier_layer_4_conv2d_3_biasadd_readvariableop_resource:g
Mlenet_traffic_sign_classifier_layer_5_conv2d_4_conv2d_readvariableop_resource:\
Nlenet_traffic_sign_classifier_layer_5_conv2d_4_biasadd_readvariableop_resource:`
Llenet_traffic_sign_classifier_layer_8_dense_1_matmul_readvariableop_resource:
??\
Mlenet_traffic_sign_classifier_layer_8_dense_1_biasadd_readvariableop_resource:	?`
Mlenet_traffic_sign_classifier_layer_10_dense_2_matmul_readvariableop_resource:	?+\
Nlenet_traffic_sign_classifier_layer_10_dense_2_biasadd_readvariableop_resource:+
identity??ELeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOp?DLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOp?CLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOp?BLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOp?ELeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOp?DLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOp?ELeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOp?DLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOp?ELeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOp?DLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOp?DLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOp?CLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOp?
BLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOpReadVariableOpKlenet_traffic_sign_classifier_layer_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02D
BLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOp?
3LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2DConv2Dlayer_1_conv2d_inputJLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
25
3LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D?
CLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOpReadVariableOpLlenet_traffic_sign_classifier_layer_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02E
CLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOp?
4LeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAddBiasAdd<LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D:output:0KLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<26
4LeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd?
1LeNet_Traffic_Sign_Classifier/layer_1_conv2d/ReluRelu=LeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<23
1LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Relu?
DLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOpReadVariableOpMlenet_traffic_sign_classifier_layer_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype02F
DLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOp?
5LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2DConv2D?LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Relu:activations:0LLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
27
5LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D?
ELeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpNlenet_traffic_sign_classifier_layer_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02G
ELeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOp?
6LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAddBiasAdd>LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D:output:0MLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<28
6LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd?
3LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/ReluRelu?LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<25
3LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Relu?
=LeNet_Traffic_Sign_Classifier/layer_3_pooling_layer_1/MaxPoolMaxPoolALeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Relu:activations:0*/
_output_shapes
:?????????<*
ksize
*
paddingVALID*
strides
2?
=LeNet_Traffic_Sign_Classifier/layer_3_pooling_layer_1/MaxPool?
DLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOpReadVariableOpMlenet_traffic_sign_classifier_layer_4_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02F
DLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOp?
5LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2DConv2DFLeNet_Traffic_Sign_Classifier/layer_3_pooling_layer_1/MaxPool:output:0LLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
27
5LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D?
ELeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNlenet_traffic_sign_classifier_layer_4_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
ELeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOp?
6LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAddBiasAdd>LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D:output:0MLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

28
6LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd?
3LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/ReluRelu?LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

25
3LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Relu?
DLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOpReadVariableOpMlenet_traffic_sign_classifier_layer_5_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02F
DLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOp?
5LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2DConv2DALeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Relu:activations:0LLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
27
5LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D?
ELeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpNlenet_traffic_sign_classifier_layer_5_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
ELeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOp?
6LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAddBiasAdd>LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D:output:0MLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????28
6LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd?
3LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/ReluRelu?LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????25
3LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Relu?
=LeNet_Traffic_Sign_Classifier/layer_6_pooling_layer_2/MaxPoolMaxPoolALeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2?
=LeNet_Traffic_Sign_Classifier/layer_6_pooling_layer_2/MaxPool?
3LeNet_Traffic_Sign_Classifier/layer_7_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  25
3LeNet_Traffic_Sign_Classifier/layer_7_flatten/Const?
5LeNet_Traffic_Sign_Classifier/layer_7_flatten/ReshapeReshapeFLeNet_Traffic_Sign_Classifier/layer_6_pooling_layer_2/MaxPool:output:0<LeNet_Traffic_Sign_Classifier/layer_7_flatten/Const:output:0*
T0*(
_output_shapes
:??????????27
5LeNet_Traffic_Sign_Classifier/layer_7_flatten/Reshape?
CLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOpReadVariableOpLlenet_traffic_sign_classifier_layer_8_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02E
CLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOp?
4LeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMulMatMul>LeNet_Traffic_Sign_Classifier/layer_7_flatten/Reshape:output:0KLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4LeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul?
DLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOpReadVariableOpMlenet_traffic_sign_classifier_layer_8_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02F
DLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOp?
5LeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAddBiasAdd>LeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul:product:0LLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5LeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd?
2LeNet_Traffic_Sign_Classifier/layer_8_dense_1/ReluRelu>LeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????24
2LeNet_Traffic_Sign_Classifier/layer_8_dense_1/Relu?
8LeNet_Traffic_Sign_Classifier/layer_9_dropout_1/IdentityIdentity@LeNet_Traffic_Sign_Classifier/layer_8_dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2:
8LeNet_Traffic_Sign_Classifier/layer_9_dropout_1/Identity?
DLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOpReadVariableOpMlenet_traffic_sign_classifier_layer_10_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02F
DLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOp?
5LeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMulMatMulALeNet_Traffic_Sign_Classifier/layer_9_dropout_1/Identity:output:0LLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+27
5LeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul?
ELeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOpReadVariableOpNlenet_traffic_sign_classifier_layer_10_dense_2_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02G
ELeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOp?
6LeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAddBiasAdd?LeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul:product:0MLeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+28
6LeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd?
6LeNet_Traffic_Sign_Classifier/layer_10_dense_2/SoftmaxSoftmax?LeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+28
6LeNet_Traffic_Sign_Classifier/layer_10_dense_2/Softmax?
IdentityIdentity@LeNet_Traffic_Sign_Classifier/layer_10_dense_2/Softmax:softmax:0F^LeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOpE^LeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOpD^LeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOpC^LeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOpF^LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOpE^LeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOpF^LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOpE^LeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOpF^LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOpE^LeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOpE^LeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOpD^LeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2?
ELeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOpELeNet_Traffic_Sign_Classifier/layer_10_dense_2/BiasAdd/ReadVariableOp2?
DLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOpDLeNet_Traffic_Sign_Classifier/layer_10_dense_2/MatMul/ReadVariableOp2?
CLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOpCLeNet_Traffic_Sign_Classifier/layer_1_conv2d/BiasAdd/ReadVariableOp2?
BLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOpBLeNet_Traffic_Sign_Classifier/layer_1_conv2d/Conv2D/ReadVariableOp2?
ELeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOpELeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/BiasAdd/ReadVariableOp2?
DLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOpDLeNet_Traffic_Sign_Classifier/layer_2_conv2d_2/Conv2D/ReadVariableOp2?
ELeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOpELeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/BiasAdd/ReadVariableOp2?
DLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOpDLeNet_Traffic_Sign_Classifier/layer_4_conv2d_3/Conv2D/ReadVariableOp2?
ELeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOpELeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/BiasAdd/ReadVariableOp2?
DLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOpDLeNet_Traffic_Sign_Classifier/layer_5_conv2d_4/Conv2D/ReadVariableOp2?
DLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOpDLeNet_Traffic_Sign_Classifier/layer_8_dense_1/BiasAdd/ReadVariableOp2?
CLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOpCLeNet_Traffic_Sign_Classifier/layer_8_dense_1/MatMul/ReadVariableOp:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?

?
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_101665

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_101396
layer_1_conv2d_input!
unknown:<
	unknown_0:<#
	unknown_1:<<
	unknown_2:<#
	unknown_3:<
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?+

unknown_10:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1008912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?
o
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_100909

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?5
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101359
layer_1_conv2d_input/
layer_1_conv2d_101324:<#
layer_1_conv2d_101326:<1
layer_2_conv2d_2_101329:<<%
layer_2_conv2d_2_101331:<1
layer_4_conv2d_3_101335:<%
layer_4_conv2d_3_101337:1
layer_5_conv2d_4_101340:%
layer_5_conv2d_4_101342:*
layer_8_dense_1_101347:
??%
layer_8_dense_1_101349:	?*
layer_10_dense_2_101353:	?+%
layer_10_dense_2_101355:+
identity??(layer_10_dense_2/StatefulPartitionedCall?&layer_1_conv2d/StatefulPartitionedCall?(layer_2_conv2d_2/StatefulPartitionedCall?(layer_4_conv2d_3/StatefulPartitionedCall?(layer_5_conv2d_4/StatefulPartitionedCall?'layer_8_dense_1/StatefulPartitionedCall?)layer_9_dropout_1/StatefulPartitionedCall?
&layer_1_conv2d/StatefulPartitionedCallStatefulPartitionedCalllayer_1_conv2d_inputlayer_1_conv2d_101324layer_1_conv2d_101326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_1009332(
&layer_1_conv2d/StatefulPartitionedCall?
(layer_2_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/layer_1_conv2d/StatefulPartitionedCall:output:0layer_2_conv2d_2_101329layer_2_conv2d_2_101331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_1009502*
(layer_2_conv2d_2/StatefulPartitionedCall?
'layer_3_pooling_layer_1/PartitionedCallPartitionedCall1layer_2_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_1008972)
'layer_3_pooling_layer_1/PartitionedCall?
(layer_4_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall0layer_3_pooling_layer_1/PartitionedCall:output:0layer_4_conv2d_3_101335layer_4_conv2d_3_101337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_1009682*
(layer_4_conv2d_3/StatefulPartitionedCall?
(layer_5_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall1layer_4_conv2d_3/StatefulPartitionedCall:output:0layer_5_conv2d_4_101340layer_5_conv2d_4_101342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_1009852*
(layer_5_conv2d_4/StatefulPartitionedCall?
'layer_6_pooling_layer_2/PartitionedCallPartitionedCall1layer_5_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_1009092)
'layer_6_pooling_layer_2/PartitionedCall?
layer_7_flatten/PartitionedCallPartitionedCall0layer_6_pooling_layer_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_1009982!
layer_7_flatten/PartitionedCall?
'layer_8_dense_1/StatefulPartitionedCallStatefulPartitionedCall(layer_7_flatten/PartitionedCall:output:0layer_8_dense_1_101347layer_8_dense_1_101349*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_1010112)
'layer_8_dense_1/StatefulPartitionedCall?
)layer_9_dropout_1/StatefulPartitionedCallStatefulPartitionedCall0layer_8_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010992+
)layer_9_dropout_1/StatefulPartitionedCall?
(layer_10_dense_2/StatefulPartitionedCallStatefulPartitionedCall2layer_9_dropout_1/StatefulPartitionedCall:output:0layer_10_dense_2_101353layer_10_dense_2_101355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_1010352*
(layer_10_dense_2/StatefulPartitionedCall?
IdentityIdentity1layer_10_dense_2/StatefulPartitionedCall:output:0)^layer_10_dense_2/StatefulPartitionedCall'^layer_1_conv2d/StatefulPartitionedCall)^layer_2_conv2d_2/StatefulPartitionedCall)^layer_4_conv2d_3/StatefulPartitionedCall)^layer_5_conv2d_4/StatefulPartitionedCall(^layer_8_dense_1/StatefulPartitionedCall*^layer_9_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2T
(layer_10_dense_2/StatefulPartitionedCall(layer_10_dense_2/StatefulPartitionedCall2P
&layer_1_conv2d/StatefulPartitionedCall&layer_1_conv2d/StatefulPartitionedCall2T
(layer_2_conv2d_2/StatefulPartitionedCall(layer_2_conv2d_2/StatefulPartitionedCall2T
(layer_4_conv2d_3/StatefulPartitionedCall(layer_4_conv2d_3/StatefulPartitionedCall2T
(layer_5_conv2d_4/StatefulPartitionedCall(layer_5_conv2d_4/StatefulPartitionedCall2R
'layer_8_dense_1/StatefulPartitionedCall'layer_8_dense_1/StatefulPartitionedCall2V
)layer_9_dropout_1/StatefulPartitionedCall)layer_9_dropout_1/StatefulPartitionedCall:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?
g
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_101649

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_101574

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
o
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_100897

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
1__inference_layer_4_conv2d_3_layer_call_fn_101623

inputs!
unknown:<
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_1009682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_101594

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?4
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101321
layer_1_conv2d_input/
layer_1_conv2d_101286:<#
layer_1_conv2d_101288:<1
layer_2_conv2d_2_101291:<<%
layer_2_conv2d_2_101293:<1
layer_4_conv2d_3_101297:<%
layer_4_conv2d_3_101299:1
layer_5_conv2d_4_101302:%
layer_5_conv2d_4_101304:*
layer_8_dense_1_101309:
??%
layer_8_dense_1_101311:	?*
layer_10_dense_2_101315:	?+%
layer_10_dense_2_101317:+
identity??(layer_10_dense_2/StatefulPartitionedCall?&layer_1_conv2d/StatefulPartitionedCall?(layer_2_conv2d_2/StatefulPartitionedCall?(layer_4_conv2d_3/StatefulPartitionedCall?(layer_5_conv2d_4/StatefulPartitionedCall?'layer_8_dense_1/StatefulPartitionedCall?
&layer_1_conv2d/StatefulPartitionedCallStatefulPartitionedCalllayer_1_conv2d_inputlayer_1_conv2d_101286layer_1_conv2d_101288*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_1009332(
&layer_1_conv2d/StatefulPartitionedCall?
(layer_2_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/layer_1_conv2d/StatefulPartitionedCall:output:0layer_2_conv2d_2_101291layer_2_conv2d_2_101293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_1009502*
(layer_2_conv2d_2/StatefulPartitionedCall?
'layer_3_pooling_layer_1/PartitionedCallPartitionedCall1layer_2_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_1008972)
'layer_3_pooling_layer_1/PartitionedCall?
(layer_4_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall0layer_3_pooling_layer_1/PartitionedCall:output:0layer_4_conv2d_3_101297layer_4_conv2d_3_101299*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_1009682*
(layer_4_conv2d_3/StatefulPartitionedCall?
(layer_5_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall1layer_4_conv2d_3/StatefulPartitionedCall:output:0layer_5_conv2d_4_101302layer_5_conv2d_4_101304*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_1009852*
(layer_5_conv2d_4/StatefulPartitionedCall?
'layer_6_pooling_layer_2/PartitionedCallPartitionedCall1layer_5_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_1009092)
'layer_6_pooling_layer_2/PartitionedCall?
layer_7_flatten/PartitionedCallPartitionedCall0layer_6_pooling_layer_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_1009982!
layer_7_flatten/PartitionedCall?
'layer_8_dense_1/StatefulPartitionedCallStatefulPartitionedCall(layer_7_flatten/PartitionedCall:output:0layer_8_dense_1_101309layer_8_dense_1_101311*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_1010112)
'layer_8_dense_1/StatefulPartitionedCall?
!layer_9_dropout_1/PartitionedCallPartitionedCall0layer_8_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010222#
!layer_9_dropout_1/PartitionedCall?
(layer_10_dense_2/StatefulPartitionedCallStatefulPartitionedCall*layer_9_dropout_1/PartitionedCall:output:0layer_10_dense_2_101315layer_10_dense_2_101317*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_1010352*
(layer_10_dense_2/StatefulPartitionedCall?
IdentityIdentity1layer_10_dense_2/StatefulPartitionedCall:output:0)^layer_10_dense_2/StatefulPartitionedCall'^layer_1_conv2d/StatefulPartitionedCall)^layer_2_conv2d_2/StatefulPartitionedCall)^layer_4_conv2d_3/StatefulPartitionedCall)^layer_5_conv2d_4/StatefulPartitionedCall(^layer_8_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2T
(layer_10_dense_2/StatefulPartitionedCall(layer_10_dense_2/StatefulPartitionedCall2P
&layer_1_conv2d/StatefulPartitionedCall&layer_1_conv2d/StatefulPartitionedCall2T
(layer_2_conv2d_2/StatefulPartitionedCall(layer_2_conv2d_2/StatefulPartitionedCall2T
(layer_4_conv2d_3/StatefulPartitionedCall(layer_4_conv2d_3/StatefulPartitionedCall2T
(layer_5_conv2d_4/StatefulPartitionedCall(layer_5_conv2d_4/StatefulPartitionedCall2R
'layer_8_dense_1/StatefulPartitionedCall'layer_8_dense_1/StatefulPartitionedCall:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?

?
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_101011

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_101035

inputs1
matmul_readvariableop_resource:	?+-
biasadd_readvariableop_resource:+
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101022

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101691

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101283
layer_1_conv2d_input!
unknown:<
	unknown_0:<#
	unknown_1:<<
	unknown_2:<#
	unknown_3:<
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?+

unknown_10:+
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_1012272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????  
.
_user_specified_namelayer_1_conv2d_input
?
N
2__inference_layer_9_dropout_1_layer_call_fn_101696

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_1010222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_layer_1_conv2d_layer_call_fn_101583

inputs!
unknown:<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_1009332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_101634

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_100968

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
0__inference_layer_8_dense_1_layer_call_fn_101674

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_1010112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101447

inputsG
-layer_1_conv2d_conv2d_readvariableop_resource:<<
.layer_1_conv2d_biasadd_readvariableop_resource:<I
/layer_2_conv2d_2_conv2d_readvariableop_resource:<<>
0layer_2_conv2d_2_biasadd_readvariableop_resource:<I
/layer_4_conv2d_3_conv2d_readvariableop_resource:<>
0layer_4_conv2d_3_biasadd_readvariableop_resource:I
/layer_5_conv2d_4_conv2d_readvariableop_resource:>
0layer_5_conv2d_4_biasadd_readvariableop_resource:B
.layer_8_dense_1_matmul_readvariableop_resource:
??>
/layer_8_dense_1_biasadd_readvariableop_resource:	?B
/layer_10_dense_2_matmul_readvariableop_resource:	?+>
0layer_10_dense_2_biasadd_readvariableop_resource:+
identity??'layer_10_dense_2/BiasAdd/ReadVariableOp?&layer_10_dense_2/MatMul/ReadVariableOp?%layer_1_conv2d/BiasAdd/ReadVariableOp?$layer_1_conv2d/Conv2D/ReadVariableOp?'layer_2_conv2d_2/BiasAdd/ReadVariableOp?&layer_2_conv2d_2/Conv2D/ReadVariableOp?'layer_4_conv2d_3/BiasAdd/ReadVariableOp?&layer_4_conv2d_3/Conv2D/ReadVariableOp?'layer_5_conv2d_4/BiasAdd/ReadVariableOp?&layer_5_conv2d_4/Conv2D/ReadVariableOp?&layer_8_dense_1/BiasAdd/ReadVariableOp?%layer_8_dense_1/MatMul/ReadVariableOp?
$layer_1_conv2d/Conv2D/ReadVariableOpReadVariableOp-layer_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02&
$layer_1_conv2d/Conv2D/ReadVariableOp?
layer_1_conv2d/Conv2DConv2Dinputs,layer_1_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
layer_1_conv2d/Conv2D?
%layer_1_conv2d/BiasAdd/ReadVariableOpReadVariableOp.layer_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02'
%layer_1_conv2d/BiasAdd/ReadVariableOp?
layer_1_conv2d/BiasAddBiasAddlayer_1_conv2d/Conv2D:output:0-layer_1_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
layer_1_conv2d/BiasAdd?
layer_1_conv2d/ReluRelulayer_1_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
layer_1_conv2d/Relu?
&layer_2_conv2d_2/Conv2D/ReadVariableOpReadVariableOp/layer_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype02(
&layer_2_conv2d_2/Conv2D/ReadVariableOp?
layer_2_conv2d_2/Conv2DConv2D!layer_1_conv2d/Relu:activations:0.layer_2_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
2
layer_2_conv2d_2/Conv2D?
'layer_2_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0layer_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02)
'layer_2_conv2d_2/BiasAdd/ReadVariableOp?
layer_2_conv2d_2/BiasAddBiasAdd layer_2_conv2d_2/Conv2D:output:0/layer_2_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
layer_2_conv2d_2/BiasAdd?
layer_2_conv2d_2/ReluRelu!layer_2_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
layer_2_conv2d_2/Relu?
layer_3_pooling_layer_1/MaxPoolMaxPool#layer_2_conv2d_2/Relu:activations:0*/
_output_shapes
:?????????<*
ksize
*
paddingVALID*
strides
2!
layer_3_pooling_layer_1/MaxPool?
&layer_4_conv2d_3/Conv2D/ReadVariableOpReadVariableOp/layer_4_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02(
&layer_4_conv2d_3/Conv2D/ReadVariableOp?
layer_4_conv2d_3/Conv2DConv2D(layer_3_pooling_layer_1/MaxPool:output:0.layer_4_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
layer_4_conv2d_3/Conv2D?
'layer_4_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0layer_4_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'layer_4_conv2d_3/BiasAdd/ReadVariableOp?
layer_4_conv2d_3/BiasAddBiasAdd layer_4_conv2d_3/Conv2D:output:0/layer_4_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
layer_4_conv2d_3/BiasAdd?
layer_4_conv2d_3/ReluRelu!layer_4_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
layer_4_conv2d_3/Relu?
&layer_5_conv2d_4/Conv2D/ReadVariableOpReadVariableOp/layer_5_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&layer_5_conv2d_4/Conv2D/ReadVariableOp?
layer_5_conv2d_4/Conv2DConv2D#layer_4_conv2d_3/Relu:activations:0.layer_5_conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
layer_5_conv2d_4/Conv2D?
'layer_5_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0layer_5_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'layer_5_conv2d_4/BiasAdd/ReadVariableOp?
layer_5_conv2d_4/BiasAddBiasAdd layer_5_conv2d_4/Conv2D:output:0/layer_5_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
layer_5_conv2d_4/BiasAdd?
layer_5_conv2d_4/ReluRelu!layer_5_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
layer_5_conv2d_4/Relu?
layer_6_pooling_layer_2/MaxPoolMaxPool#layer_5_conv2d_4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2!
layer_6_pooling_layer_2/MaxPool
layer_7_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
layer_7_flatten/Const?
layer_7_flatten/ReshapeReshape(layer_6_pooling_layer_2/MaxPool:output:0layer_7_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
layer_7_flatten/Reshape?
%layer_8_dense_1/MatMul/ReadVariableOpReadVariableOp.layer_8_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%layer_8_dense_1/MatMul/ReadVariableOp?
layer_8_dense_1/MatMulMatMul layer_7_flatten/Reshape:output:0-layer_8_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/MatMul?
&layer_8_dense_1/BiasAdd/ReadVariableOpReadVariableOp/layer_8_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_8_dense_1/BiasAdd/ReadVariableOp?
layer_8_dense_1/BiasAddBiasAdd layer_8_dense_1/MatMul:product:0.layer_8_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/BiasAdd?
layer_8_dense_1/ReluRelu layer_8_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_8_dense_1/Relu?
layer_9_dropout_1/IdentityIdentity"layer_8_dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
layer_9_dropout_1/Identity?
&layer_10_dense_2/MatMul/ReadVariableOpReadVariableOp/layer_10_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?+*
dtype02(
&layer_10_dense_2/MatMul/ReadVariableOp?
layer_10_dense_2/MatMulMatMul#layer_9_dropout_1/Identity:output:0.layer_10_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/MatMul?
'layer_10_dense_2/BiasAdd/ReadVariableOpReadVariableOp0layer_10_dense_2_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02)
'layer_10_dense_2/BiasAdd/ReadVariableOp?
layer_10_dense_2/BiasAddBiasAdd!layer_10_dense_2/MatMul:product:0/layer_10_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/BiasAdd?
layer_10_dense_2/SoftmaxSoftmax!layer_10_dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
layer_10_dense_2/Softmax?
IdentityIdentity"layer_10_dense_2/Softmax:softmax:0(^layer_10_dense_2/BiasAdd/ReadVariableOp'^layer_10_dense_2/MatMul/ReadVariableOp&^layer_1_conv2d/BiasAdd/ReadVariableOp%^layer_1_conv2d/Conv2D/ReadVariableOp(^layer_2_conv2d_2/BiasAdd/ReadVariableOp'^layer_2_conv2d_2/Conv2D/ReadVariableOp(^layer_4_conv2d_3/BiasAdd/ReadVariableOp'^layer_4_conv2d_3/Conv2D/ReadVariableOp(^layer_5_conv2d_4/BiasAdd/ReadVariableOp'^layer_5_conv2d_4/Conv2D/ReadVariableOp'^layer_8_dense_1/BiasAdd/ReadVariableOp&^layer_8_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2R
'layer_10_dense_2/BiasAdd/ReadVariableOp'layer_10_dense_2/BiasAdd/ReadVariableOp2P
&layer_10_dense_2/MatMul/ReadVariableOp&layer_10_dense_2/MatMul/ReadVariableOp2N
%layer_1_conv2d/BiasAdd/ReadVariableOp%layer_1_conv2d/BiasAdd/ReadVariableOp2L
$layer_1_conv2d/Conv2D/ReadVariableOp$layer_1_conv2d/Conv2D/ReadVariableOp2R
'layer_2_conv2d_2/BiasAdd/ReadVariableOp'layer_2_conv2d_2/BiasAdd/ReadVariableOp2P
&layer_2_conv2d_2/Conv2D/ReadVariableOp&layer_2_conv2d_2/Conv2D/ReadVariableOp2R
'layer_4_conv2d_3/BiasAdd/ReadVariableOp'layer_4_conv2d_3/BiasAdd/ReadVariableOp2P
&layer_4_conv2d_3/Conv2D/ReadVariableOp&layer_4_conv2d_3/Conv2D/ReadVariableOp2R
'layer_5_conv2d_4/BiasAdd/ReadVariableOp'layer_5_conv2d_4/BiasAdd/ReadVariableOp2P
&layer_5_conv2d_4/Conv2D/ReadVariableOp&layer_5_conv2d_4/Conv2D/ReadVariableOp2P
&layer_8_dense_1/BiasAdd/ReadVariableOp&layer_8_dense_1/BiasAdd/ReadVariableOp2N
%layer_8_dense_1/MatMul/ReadVariableOp%layer_8_dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
]
layer_1_conv2d_inputE
&serving_default_layer_1_conv2d_input:0?????????  D
layer_10_dense_20
StatefulPartitionedCall:0?????????+tensorflow/serving/predict:??
?]
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?Y
_tf_keras_sequential?Y{"name": "LeNet_Traffic_Sign_Classifier", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "LeNet_Traffic_Sign_Classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_1_conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "layer_1_conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "layer_2_conv2d_2", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_3_pooling_layer_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer_4_conv2d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "layer_5_conv2d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_6_pooling_layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "layer_7_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "layer_8_dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "layer_9_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "layer_10_dense_2", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 32, 1]}, "float32", "layer_1_conv2d_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "LeNet_Traffic_Sign_Classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_1_conv2d_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "layer_1_conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "layer_2_conv2d_2", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "layer_3_pooling_layer_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "layer_4_conv2d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "layer_5_conv2d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "MaxPooling2D", "config": {"name": "layer_6_pooling_layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 14}, {"class_name": "Flatten", "config": {"name": "layer_7_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "layer_8_dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "Dropout", "config": {"name": "layer_9_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "layer_10_dense_2", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 25}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "layer_1_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer_1_conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "layer_2_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer_2_conv2d_2", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 60}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 60]}}
?
	variables
trainable_variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_3_pooling_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "layer_3_pooling_layer_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 27}}
?


!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "layer_4_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer_4_conv2d_3", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 60}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 60]}}
?


'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "layer_5_conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer_5_conv2d_4", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 30]}}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_6_pooling_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "layer_6_pooling_layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_7_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "layer_7_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 31}}
?

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_8_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "layer_8_dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 480}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 480]}}
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_9_dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "layer_9_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 19}
?

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_10_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "layer_10_dense_2", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
?
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratem?m?m?m?!m?"m?'m?(m?5m?6m??m?@m?v?v?v?v?!v?"v?'v?(v?5v?6v??v?@v?"
	optimizer
v
0
1
2
3
!4
"5
'6
(7
58
69
?10
@11"
trackable_list_wrapper
v
0
1
2
3
!4
"5
'6
(7
58
69
?10
@11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
Lmetrics
regularization_losses

Mlayers
Nnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
/:-<2layer_1_conv2d/kernel
!:<2layer_1_conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Olayer_regularization_losses
Player_metrics

Qlayers
trainable_variables
regularization_losses
Rmetrics
Snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/<<2layer_2_conv2d_2/kernel
#:!<2layer_2_conv2d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Tlayer_regularization_losses
Ulayer_metrics

Vlayers
trainable_variables
regularization_losses
Wmetrics
Xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Ylayer_regularization_losses
Zlayer_metrics

[layers
trainable_variables
regularization_losses
\metrics
]non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/<2layer_4_conv2d_3/kernel
#:!2layer_4_conv2d_3/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#	variables
^layer_regularization_losses
_layer_metrics

`layers
$trainable_variables
%regularization_losses
ametrics
bnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/2layer_5_conv2d_4/kernel
#:!2layer_5_conv2d_4/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)	variables
clayer_regularization_losses
dlayer_metrics

elayers
*trainable_variables
+regularization_losses
fmetrics
gnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-	variables
hlayer_regularization_losses
ilayer_metrics

jlayers
.trainable_variables
/regularization_losses
kmetrics
lnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
mlayer_regularization_losses
nlayer_metrics

olayers
2trainable_variables
3regularization_losses
pmetrics
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
??2layer_8_dense_1/kernel
#:!?2layer_8_dense_1/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables
rlayer_regularization_losses
slayer_metrics

tlayers
8trainable_variables
9regularization_losses
umetrics
vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
wlayer_regularization_losses
xlayer_metrics

ylayers
<trainable_variables
=regularization_losses
zmetrics
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(	?+2layer_10_dense_2/kernel
#:!+2layer_10_dense_2/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
A	variables
|layer_regularization_losses
}layer_metrics

~layers
Btrainable_variables
Cregularization_losses
metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 34}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 25}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
4:2<2Adam/layer_1_conv2d/kernel/m
&:$<2Adam/layer_1_conv2d/bias/m
6:4<<2Adam/layer_2_conv2d_2/kernel/m
(:&<2Adam/layer_2_conv2d_2/bias/m
6:4<2Adam/layer_4_conv2d_3/kernel/m
(:&2Adam/layer_4_conv2d_3/bias/m
6:42Adam/layer_5_conv2d_4/kernel/m
(:&2Adam/layer_5_conv2d_4/bias/m
/:-
??2Adam/layer_8_dense_1/kernel/m
(:&?2Adam/layer_8_dense_1/bias/m
/:-	?+2Adam/layer_10_dense_2/kernel/m
(:&+2Adam/layer_10_dense_2/bias/m
4:2<2Adam/layer_1_conv2d/kernel/v
&:$<2Adam/layer_1_conv2d/bias/v
6:4<<2Adam/layer_2_conv2d_2/kernel/v
(:&<2Adam/layer_2_conv2d_2/bias/v
6:4<2Adam/layer_4_conv2d_3/kernel/v
(:&2Adam/layer_4_conv2d_3/bias/v
6:42Adam/layer_5_conv2d_4/kernel/v
(:&2Adam/layer_5_conv2d_4/bias/v
/:-
??2Adam/layer_8_dense_1/kernel/v
(:&?2Adam/layer_8_dense_1/bias/v
/:-	?+2Adam/layer_10_dense_2/kernel/v
(:&+2Adam/layer_10_dense_2/bias/v
?2?
!__inference__wrapped_model_100891?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *;?8
6?3
layer_1_conv2d_input?????????  
?2?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101447
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101505
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101321
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101359?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101069
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101534
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101563
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101283?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_101574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_layer_1_conv2d_layer_call_fn_101583?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_101594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_layer_2_conv2d_2_layer_call_fn_101603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_100897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
8__inference_layer_3_pooling_layer_1_layer_call_fn_100903?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_101614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_layer_4_conv2d_3_layer_call_fn_101623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_101634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_layer_5_conv2d_4_layer_call_fn_101643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_100909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
8__inference_layer_6_pooling_layer_2_layer_call_fn_100915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_101649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_layer_7_flatten_layer_call_fn_101654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_101665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_layer_8_dense_1_layer_call_fn_101674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101679
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101691?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_layer_9_dropout_1_layer_call_fn_101696
2__inference_layer_9_dropout_1_layer_call_fn_101701?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_101712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_layer_10_dense_2_layer_call_fn_101721?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_101396layer_1_conv2d_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101321?!"'(56?@M?J
C?@
6?3
layer_1_conv2d_input?????????  
p 

 
? "%?"
?
0?????????+
? ?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101359?!"'(56?@M?J
C?@
6?3
layer_1_conv2d_input?????????  
p

 
? "%?"
?
0?????????+
? ?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101447v!"'(56?@??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????+
? ?
Y__inference_LeNet_Traffic_Sign_Classifier_layer_call_and_return_conditional_losses_101505v!"'(56?@??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????+
? ?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101069w!"'(56?@M?J
C?@
6?3
layer_1_conv2d_input?????????  
p 

 
? "??????????+?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101283w!"'(56?@M?J
C?@
6?3
layer_1_conv2d_input?????????  
p

 
? "??????????+?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101534i!"'(56?@??<
5?2
(?%
inputs?????????  
p 

 
? "??????????+?
>__inference_LeNet_Traffic_Sign_Classifier_layer_call_fn_101563i!"'(56?@??<
5?2
(?%
inputs?????????  
p

 
? "??????????+?
!__inference__wrapped_model_100891?!"'(56?@E?B
;?8
6?3
layer_1_conv2d_input?????????  
? "C?@
>
layer_10_dense_2*?'
layer_10_dense_2?????????+?
L__inference_layer_10_dense_2_layer_call_and_return_conditional_losses_101712]?@0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????+
? ?
1__inference_layer_10_dense_2_layer_call_fn_101721P?@0?-
&?#
!?
inputs??????????
? "??????????+?
J__inference_layer_1_conv2d_layer_call_and_return_conditional_losses_101574l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????<
? ?
/__inference_layer_1_conv2d_layer_call_fn_101583_7?4
-?*
(?%
inputs?????????  
? " ??????????<?
L__inference_layer_2_conv2d_2_layer_call_and_return_conditional_losses_101594l7?4
-?*
(?%
inputs?????????<
? "-?*
#? 
0?????????<
? ?
1__inference_layer_2_conv2d_2_layer_call_fn_101603_7?4
-?*
(?%
inputs?????????<
? " ??????????<?
S__inference_layer_3_pooling_layer_1_layer_call_and_return_conditional_losses_100897?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
8__inference_layer_3_pooling_layer_1_layer_call_fn_100903?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_layer_4_conv2d_3_layer_call_and_return_conditional_losses_101614l!"7?4
-?*
(?%
inputs?????????<
? "-?*
#? 
0?????????


? ?
1__inference_layer_4_conv2d_3_layer_call_fn_101623_!"7?4
-?*
(?%
inputs?????????<
? " ??????????

?
L__inference_layer_5_conv2d_4_layer_call_and_return_conditional_losses_101634l'(7?4
-?*
(?%
inputs?????????


? "-?*
#? 
0?????????
? ?
1__inference_layer_5_conv2d_4_layer_call_fn_101643_'(7?4
-?*
(?%
inputs?????????


? " ???????????
S__inference_layer_6_pooling_layer_2_layer_call_and_return_conditional_losses_100909?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
8__inference_layer_6_pooling_layer_2_layer_call_fn_100915?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_layer_7_flatten_layer_call_and_return_conditional_losses_101649a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
0__inference_layer_7_flatten_layer_call_fn_101654T7?4
-?*
(?%
inputs?????????
? "????????????
K__inference_layer_8_dense_1_layer_call_and_return_conditional_losses_101665^560?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_layer_8_dense_1_layer_call_fn_101674Q560?-
&?#
!?
inputs??????????
? "????????????
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101679^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
M__inference_layer_9_dropout_1_layer_call_and_return_conditional_losses_101691^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
2__inference_layer_9_dropout_1_layer_call_fn_101696Q4?1
*?'
!?
inputs??????????
p 
? "????????????
2__inference_layer_9_dropout_1_layer_call_fn_101701Q4?1
*?'
!?
inputs??????????
p
? "????????????
$__inference_signature_wrapper_101396?!"'(56?@]?Z
? 
S?P
N
layer_1_conv2d_input6?3
layer_1_conv2d_input?????????  "C?@
>
layer_10_dense_2*?'
layer_10_dense_2?????????+