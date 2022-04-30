

import * as tfa from '@tensorflow/tfjs';

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
import * as tfgpu from  '@tensorflow/tfjs-node-gpu';
const tf = tfgpu.default;

import * as CRa from "./chainReact.mjs" 
const CR = CRa.default;


const tickBlank = [];
for( var n = 0; n < 134; n++ ) tickBlank.push( 0 );
const boardBlank = [];
for( var n = 0; n < 40*7; n++ ) boardBlank.push( 0 );
const allowBlank = [];
for( var n = 0; n < 40; n++ ) allowBlank.push( 0 );
const moveBlank = [];
for( var n = 0; n < 40; n++ ) moveBlank.push( 0 );
const maskBlank = [];
for( var n = 0; n < 134; n++ ) maskBlank.push( 0 );

async function trainOneGame( model, batch, n ) {
	var gameState;
	var batchTicks = [];
	var batchBoards = [];
	var batchAllows = [];
	var batchMoves = [];
	var batchMasks = [];

	console.log( "Play games");
	for( var job = 0; job < batch; job++ ) {
		while( !(gameState = CR.runGame() ) );

		var ticks = [];
		batchTicks.push( ticks );
		var boards = [];
		batchBoards.push( boards );
		var allows = [];
		batchAllows.push( allows );
		var moves = [];
		batchMoves.push( moves );
		var masks = [];
		batchMasks.push( masks );
		var step;
		for( step = 0; step < gameState.length; step++ ) {
			ticks.push( gameState[step].tick );
			boards.push( gameState[step].board );
			allows.push( gameState[step].canMove );
			moves.push( gameState[step].lastMove );
			masks.push( [0.0] );
		}
		for( ;step < 134; step++ ) {
			ticks.push( tickBlank );
			boards.push( boardBlank );
			allows.push( allowBlank );
			moves.push( moveBlank );
			masks.push( [1.0] );
		}
	}
	console.log( "Gathered games");
	{
		var tickTensor = tf.tensor3d( batchTicks );
		//var tickTensorBatch = tf.tensor2d( [tickTensor], [1,tickTensor.shape[0]] )
		// 160 bits

		var boardTensor = tf.tensor3d( batchBoards );
		// 40 * 7
		var canMoveTensor = tf.tensor3d( batchAllows );
		// 40 bits (mask on softmax activation)
		var lastMoveTensor = tf.tensor3d( batchMoves );
		var maskTensor = tf.tensor3d( batchMasks );
		// 40 bits (target)
		console.log( "training...");

		//tf.train.latest_checkpoint(checkpoint_dir)
		
		var result = await model.trainOnBatch( [tickTensor, boardTensor, maskTensor], lastMoveTensor );
		console.log( "result:" + result );
		//result.save( 'file://./crtrain-'+n );
		
		const saveResults = await model.save('file://./cr-' + n );

		boardTensor.dispose();
		tickTensor.dispose();
		canMoveTensor.dispose();
		lastMoveTensor.dispose();
		maskTensor.dispose();
	}
}


async function test3() {

	const batchSize= 100;

	const boardInput = tf.input({		
		name : "board",  
		shape:[134,40*7]
	});

	const tickMask = tf.input( {
		name : "tick",
		shape : [134,134],
	})

	const timeStepMask = tf.input( {
		name : "skipTick",
		shape : [134,1],
	})
	var maskLayer = tf.layers.masking();
	var mask = maskLayer.apply( timeStepMask );

	var concatLayer = tf.layers.concatenate( )
	var merge = concatLayer.apply([tickMask, boardInput,mask]);
	console.log( "merge:" + JSON.stringify(merge.shape));


	// masking input...
	const simpleRNNConfig = {
		name : 'theBrain',
		units : 64,
		activation : "relu", // 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh' | null
		useBias : true,
		kernelInializer : 'randomNomral', // 'constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string|tf.initializers.Initializer)
		recurrentInitializer : 'randomNormal',
		biasInitializer : 'randomNormal',

		//kernelRegularizer : , // 'l1l2'|string|Regularizer
		//recurrentRegularizer 
		//biasRegularizer 
		//batchInputShape : merge.shape,
		kernelConstraint : 'unitNorm', // 'maxNorm'|'minMaxNorm'|'nonNeg'|'unitNorm'|string|tf.constraints.Constraint
		recurrentConstraint : 'unitNorm',
		biasConstraint : 'unitNorm',

		dropout : 0.10,
		recurrentDropout : 0.08,
		//cell : ,

		returnSequences : true, // last output or whole sequence?
		returnState : false, // 
		goBackwards : false, // regeneration?
		stateful : false, // data across input is realted (FALSE!)
	}

	const simpleRNNReaderConfig = {
		name : 'theBrain',
		units : 64,
		activation : "relu", // 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh' | null
		useBias : true,
		kernelInializer : 'randomNomral', // 'constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string|tf.initializers.Initializer)
		recurrentInitializer : 'randomNormal',
		biasInitializer : 'randomNormal',

		//kernelRegularizer : , // 'l1l2'|string|Regularizer
		//recurrentRegularizer 
		//biasRegularizer 
		//batchInputShape : merge.shape,
		kernelConstraint : 'unitNorm', // 'maxNorm'|'minMaxNorm'|'nonNeg'|'unitNorm'|string|tf.constraints.Constraint
		recurrentConstraint : 'unitNorm',
		biasConstraint : 'unitNorm',

		dropout : 0.10,
		recurrentDropout : 0.08,
		//cell : ,

		returnSequences : false, // last output or whole sequence?
		returnState : false, // 
		goBackwards : false, // regeneration?
		stateful : true, // data across input is realted (FALSE!)

	}


	const theBrainLayer = tf.layers.simpleRNN( simpleRNNConfig );
	const theBrain = theBrainLayer.apply( merge );

	simpleRNNConfig.name = "theBrain2";
	const theBrainLayer2 = tf.layers.simpleRNN( simpleRNNConfig );
	//const theBrain2 = theBrainLayer2.apply( bl0bl1IInterface );
	const bl0bl1Interface = tf.layers.bidirectional( { name:"b0b1", 
        		layer : theBrainLayer2,
        		mergeMode: 'sum'  // mul, concat, ave
		} ); 
	const bl0bl1 = bl0bl1Interface	.apply( theBrain );

	simpleRNNConfig.name = "theBrain3";
	const theBrainLayer3 = tf.layers.simpleRNN( simpleRNNConfig );
	//const theBrain3 = theBrainLayer3.apply( bl1bl2Interface );

	const bl1bl2Interface = tf.layers.bidirectional( { name:"b1b2"
        		, layer : theBrainLayer3
        		, mergeMode: 'sum'  // mul, concat, ave
		} ); 

	const brainOutput = bl1bl2Interface.apply( bl0bl1 );
        
	//const theBrainOut = theBrain[0];
	//const theBrainState = theBrain[1];

	console.log( "brainshap1 : " + JSON.stringify( brainOutput.shape ) );
	//console.log( "brainshap1 : " + JSON.stringify( theBrain[0].shape ) );
	//console.log( "brainshap2 : " + JSON.stringify( theBrain[1].shape ) );
	console.log( "brain(merge)" + theBrainLayer.computeOutputShape(merge.shape));


	const moveChoicesLayer = tf.layers.dense( 
		{ 
			name : "weighChoices",
			units : 40,
			activation: "relu"
		 }
	)

	const timeDistribution = tf.layers.timeDistributed({
		name: "timeMuxer",
		layer: moveChoicesLayer
	});
	
	const timeDistributor = timeDistribution.apply( brainOutput );

	//const moveChoices = moveChoicesLayer.apply( timeDistributor );

	console.log( "choiceWeigher : " + JSON.stringify( timeDistributor.shape ) );

	var moveChooser = tf.layers.softmax({
		name: "moveChooser",
		axis : -1, // the last one
	});
	const moveChoice = moveChooser.apply(timeDistributor);

	console.log( "choiceWeigher : " + JSON.stringify( moveChoice.shape ) );

	const optimizer = tf.train.adam( 0.1 );

	const model = tf.model( {
		inputs: [tickMask,boardInput,timeStepMask],
		outputs:moveChoice,
	} );


	await model.compile( {
		optimizer : optimizer,
		loss : 'categoricalCrossentropy',
	} );

	for( var n = 0; n < 10; n++ )
		await trainOneGame( model, batchSize, n );

	
	CR.initBoard();
	var index = CR.dumpGame();
	var log = CR.getGameLog();
	{
		var tickTensor = tf.tensor3d( [[log[index].tick]] );
		//var tickTensorBatch = tf.tensor2d( [tickTensor], [1,tickTensor.shape[0]] )
		// 160 bits

		var boardTensor = tf.tensor3d( [[log[index].board]] );
		// 40 * 7
		var canMoveTensor = tf.tensor3d( [[log[index].canMove]] );
		// 40 bits (mask on softmax activation)
		var lastMoveTensor = tf.tensor3d( [[log[index].lastMove]] );
		var maskTensor = tf.ones([1,1,1]);

		const boardInput = tf.input({		
			name : "board",  
			batchShape:[1,1,40*7]
		});
	
		const tickMask = tf.input( {
			name : "tick",
			batchShape : [1,1,134],
		})

		const timeStepMask = tf.input( {
			name : "skipTick",
			batchShape : [1,1,1],
		})
	
		var concatLayer2 = tf.layers.concatenate( )
		var merge = concatLayer2.apply([tickMask, boardInput, timeStepMask]);
		//const timeDistribution = tf.layers.timeDistributed({
	//		name: "timeMuxer",
	//		layer: concatLayer2
	//	});
	//	var mergeSplitter = timeDistribution.apply( merge );
		simpleRNNReaderConfig.name = "theBrain2";
		const readerBrainLayer = tf.layers.simpleRNN( simpleRNNReaderConfig );
		const theBrain = readerBrainLayer.apply( merge );

		simpleRNNReaderConfig.name = "theBrain3";
		const theBrainLayer2 = tf.layers.simpleRNN( simpleRNNReaderConfig );
		const theBrain2 = theBrainLayer2.apply( theBrain );
	
		const theBrainLayer3 = tf.layers.simpleRNN( simpleRNNReaderConfig );
		const theBrain3 = theBrainLayer3.apply( theBrain2 );
	

		const moveChoicesLayer = tf.layers.dense( 
			{ 
				name : "weighChoices",
				units : 40,
				activation: "relu"
			 }
		)
		const moveChoices = moveChoicesLayer.apply( theBrain3 );
	

		var moveChooser = tf.layers.softmax({
			name: "moveChooser",
			axis : -1, // the last one
		});

		const moveChoice = moveChooser.apply(moveChoices);
	
		const predmodel = tf.model( {
			inputs: [tickMask,boardInput, timeStepMask],
			outputs:moveChoice,
		} );

		{
			var layermap = [ [0,0],[1,1],[2,2],[3,3],[4,4],[5,4],[6,5],[7,6] ];
			//model.layers[0].getWeights()[0].print();
			model.layers.forEach( (layer,idx)=>{
				if( layer.weights.length ) {
					console.log( layer.name+":"+layer.getWeights() );
					predmodel.layers[idx-1].setWeights( layer.getWeights() );
				}
			})
			var thing = model.layers[0].getWeights();
		}

		await predmodel.save( "file://./playBrain" );

		//const loadedModel = await tf.loadLayersModel('file://./cr-1', {} );
//		const weights = await tf.io.loadWeights( 'file://./cr-1/weights.bin');
		var prediction = await predmodel.predict( [tickTensor,boardTensor,maskTensor] );
		prediction.print();
		var thing = tf.argMax(prediction, 2).dataSync();
		console.log( "GO:"+ thing[0] );

		var prediction = await predmodel.predict( [tickTensor,boardTensor,maskTensor] );
		prediction.print();
		var thing = tf.argMax(prediction, 2).dataSync();
		console.log( "GO:"+ thing[0] );
		var prediction = await predmodel.predict( [tickTensor,boardTensor,maskTensor] );
		prediction.print();
		var thing = tf.argMax(prediction, 2).dataSync();
		console.log( "GO:"+ thing[0] );
		var prediction = await predmodel.predict( [tickTensor,boardTensor,maskTensor] );
		prediction.print();
		var thing = tf.argMax(prediction, 2).dataSync();

		CR.putPiece( thing%8, (thing / 8)|0 );

		console.log( "GO:"+ thing[0] );
		boardTensor.dispose();
		tickTensor.dispose();
		canMoveTensor.dispose();
		lastMoveTensor.dispose();
		maskTensor.dispose();
	}
	//const saveResults = await model.save('file://./my-model-1')


	/*
	var lstm_cpn = tf.layers.lstm( simpleRNNConfig );
	lstm_cpn.apply( merge );
	//model.add( merge );
	console.log( JSON.stringify( lstm_cpn.shape ) );
	console.log( lstm_cpn.computeOutputShape(merge.shape));

	var gru_cpn = tf.layers.gru( simpleRNNConfig );
	gru_cpn.apply( merge );
	//model.add( merge );
	console.log( JSON.stringify( gru_cpn.shape ) );
	console.log( gru_cpn.computeOutputShape(merge.shape));

	var gru_cpn2 = tf.layers.gru( simpleRNNConfig );
	gru_cpn2.apply( gru_cpn );
	//model.add( merge );
	console.log( JSON.stringify( gru_cpn2.shape ) );
	console.log( gru_cpn2.computeOutputShape(gru_cpn.shape));
*/
	/*
	const moveFilter = tf.layers.masking ( {
		maskValue : 0,
	})
	*/

}
test3()

async function test2() {
	const model = tf.sequential(); // sequential model?  stack with no branches
        
        model.add( tf.layers.dense( {units:1, inputShape: [1]}));
        model.compile( {
        	loss: "meanSquaredError",
                optimizer: 'sgd',  // stocastic gradient descent
        } );
        
        const xs = tf.tensor2d( [-1,0,1,2,3,4], [6, 1] );
        const ys = tf.tensor2d( [-3,-1,1,3,5,7], [6, 1] );
        
        await model.fit( xs, ys, { epochs : 500
		,   callbacks: {
		    onEpochEnd: (epoch, log) => {
			//console.log(`Epoch ${epoch}: loss = ${log.loss}`)
		    }
		}
	 } );
		        console.log( ''+model.predict( tf.tensor2d( [10], [1, 1] ) ));
		        console.log( model.predict( tf.tensor2d( [55], [1, 1] ) ));
}

//test2();        


async function testIris( xTrain, yTrain, xTest, yTest ) {
	const model = tf.sequential(); // sequential model?  stack with no branches
        
        model.add( 
		tf.layers.dense( {
				units:10
				,activation:'sigmoid'
				, inputShape: [xTrain.shape[1] ]
			}
		)
	);

        model.add( 
		tf.layers.dense( {
				units:3
				,activation:'softmax'
				, inputShape: [xTrain.shape[1] ]
			}
		)
	);
	const learningRate = 0.01;
	const optimizer = tf.train.adam(learningRate) ;
        model.compile( {
        	loss: "categoricalCrossentropy",
                optimizer: optimizer,  // stocastic gradient descent
		metrics:['accuracy'],
        } );
        
        const xs = tf.tensor2d( [-1,0,1,2,3,4], [6, 1] );
        const ys = tf.tensor2d( [-3,-1,1,3,5,7], [6, 1] );
        
        await model.fit( xTrain, yTrain, { epochs : 500
		, validationData : [xTest, yTest ] 
		,   callbacks: {
		    onEpochEnd: (epoch, log) => {
			//console.log(`Epoch ${epoch}: loss = ${log.loss}`)
			//await tf.nextFame();
		    }
		}
	 } );

	return model;
		        console.log( ''+model.predict( tf.tensor2d( [10], [1, 1] ) ));
		        console.log( model.predict( tf.tensor2d( [55], [1, 1] ) ));
}


/*
// Train a simple model:
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
	}
);


  */