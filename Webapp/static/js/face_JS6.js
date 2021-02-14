# Models 


let model = null;
let isFrame = false;

let keypoints_dict = {
    'upper_lip' : [61, 39, 0, 269],
    'lower_lip' : [291, 181, 17, 405],
    'upper_righteye' : [33, 160, 159, 158],
    'lower_righteye' : [133, 144, 145, 153],
    'upper_lefteye' : [263, 387, 386, 385],
    'lower_lefteye' : [362, 373, 374, 380],
    'iris_righteye' : [473, 474, 475, 476, 477],
    'iris_lefteye' : [468, 469, 470, 471, 472]
    // right_iris_0 ==> center, iris_1 ==> right_end, iris_2 ==> top, iris_3 ==> left_end, iris_4 ==> bottom
    // left_iris_0 ==> center, iris_3 ==> right_end, iris_2 ==> top, iris_1 ==> left_end, iris_4 ==> bottom
};

let states2ind = {
    'neutral': 0,
    'alert': 1, 
    'drowsy': 2
};

let ind2states = {
    0: 'neutral',
    1: 'alert',
    2: 'drowsy'
};

let gazeStates2ind = {
    'center': 0,
    'right' : 1, 
    'up': 2,
    'left': 3,
    'down': 4
};

let ind2gazeStates = {
    0: 'center',
    1: 'right', 
    2: 'up',
    3: 'left',
    4: 'down'
};


let video = null;
let canvas = null;
let context = null;


let ears_norm = [0, 1];
let mars_norm = [0, 1];
let moes_norm = [0, 1];
let calib_frames = 50;

let ear_main = -1000;
let mar_main = -1000;
let moe_main = -1000;

let state_para = null;
let gaze_para = null;
let result_state = ind2states[0];
let result_gaze = ind2gazeStates[0];

let timer = null;

let gaze_left_threshold = 0.40;
let gaze_right_threshold = 0.60;
let gaze_frames_threshold = 7;
let state_frames_threshold = 10;
let drowsy_threshold = 4;
let alert_threshold = 0.5;


function startCamera(){
    if (navigator.mediaDevices.getUserMedia) {
        video = document.getElementById("webStream");
        navigator.mediaDevices.getUserMedia({ video: true }).then(
            function (stream) {
            video.srcObject = stream;
            canvas = document.getElementById("outputCanvas"); 
            context = canvas.getContext("2d");
          }).catch(function (err0r) {
            console.log("Something went wrong!");
            console.log(err0r)
          });
      }
}

function logPoints(mesh)
{
    for (let key in keypoints_dict)
    {
        console.log(key);
        points = keypoints_dict[key];
        console.log(points);
        for (let j=0; j<points.length; j++)
        {
            xdp = mesh[points[j]][0];
            ydp = mesh[points[j]][1];
            console.log(points[j] + " " + xdp + " "  + ydp + " " + key);
        }
    }
}

function drawPoints(mesh)
{
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.beginPath();
    for (let key in keypoints_dict)
    {
        points = keypoints_dict[key];
        for (let j=0; j<points.length; j++)
        {
            x = mesh[points[j]][0];
            y = mesh[points[j]][1];
            context.beginPath();
            context.arc(x, y, 1 /* radius */, 0, 3 * Math.PI);
            context.fillStyle = "aqua";
            context.fill();
        }
    }
}

function drawAllPoints(mesh)
{  
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.beginPath();
    for(let i=0; i<mesh.length; i++)
    {
        const x = mesh[i][0];
        const y = mesh[i][1];
        context.beginPath();
        context.arc(x, y, 1 /* radius */, 0, 3 * Math.PI);
        context.fillStyle = "aqua";
        context.fill();
    }
}



function distance(p1, p2)
{
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(0.5);
}

function eye_aspect_ratio(mesh, upper_eye_points, lower_eye_points)
{
    N1 = distance(mesh[upper_eye_points[1]], mesh[lower_eye_points[1]]);
    N2 = distance(mesh[upper_eye_points[2]], mesh[lower_eye_points[2]]);
    N3 = distance(mesh[upper_eye_points[3]], mesh[lower_eye_points[3]]);
    D = distance(mesh[upper_eye_points[0]], mesh[lower_eye_points[0]]);
    return (N1 + N2 + N3)/(3*D);
}

function eye_feature(mesh)
{
    return (eye_aspect_ratio(mesh, keypoints_dict['upper_righteye'], keypoints_dict['lower_righteye']) + 
    eye_aspect_ratio(mesh, keypoints_dict['upper_lefteye'], keypoints_dict['lower_lefteye']))/2;
}

function mouth_feature(mesh)
{
    N1 = distance(mesh[keypoints_dict['upper_lip'][1]], mesh[keypoints_dict['lower_lip'][1]]);
    N2 = distance(mesh[keypoints_dict['upper_lip'][2]], mesh[keypoints_dict['lower_lip'][2]]);
    N3 = distance(mesh[keypoints_dict['upper_lip'][3]], mesh[keypoints_dict['lower_lip'][3]]);
    D = distance(mesh[keypoints_dict['upper_lip'][0]], mesh[keypoints_dict['lower_lip'][0]]);
   return (N1 + N2 + N3)/(3*D);
}

function mouth_over_eye_feature(mouth_aspect_ratio, eye_aspect_ratio)
{
    return mouth_aspect_ratio/eye_aspect_ratio;
}

function average(data)
{
    var sum = data.reduce(function(sum, value)
    {
      return sum + value;
    }, 0);
  
    var avg = sum / data.length;
    return avg;
}

function standardDeviation(values)
{
    var avg = average(values);
    
    var squareDiffs = values.map(function(value)
    {
      var diff = value - avg;
      var sqrDiff = diff * diff;
      return sqrDiff;
    });
    
    var avgSquareDiff = average(squareDiffs);
  
    var stdDev = Math.sqrt(avgSquareDiff);
    return stdDev;
}


function eye_lr(mesh)
{
    right_iris = (mesh[keypoints_dict['iris_righteye'][0]][0] - mesh[keypoints_dict['upper_righteye'][0]][0])
    right_width = distance(mesh[keypoints_dict['upper_righteye'][0]], mesh[keypoints_dict['lower_righteye'][0]])
    right_distance = right_iris/right_width

    left_iris = (mesh[keypoints_dict['iris_lefteye'][0]][0] - mesh[keypoints_dict['lower_lefteye'][0]][0])
    left_width = distance(mesh[keypoints_dict['upper_lefteye'][0]], mesh[keypoints_dict['lower_lefteye'][0]])
    left_distance = left_iris/left_width

    return (right_distance + left_distance)/2;
}

async function startModel(){
    console.log('starting Model');
    isFrame = true;
    let decay = 0.9;
    let frame_count = 0;
    let old_state = 0;
    let new_state = 0;
    let new_frame_count = 0;

    let gaze_frame_count = 0;
    let gaze_old_state = 0;
    let gaze_new_state = 0;
    let gaze_new_frame_count = 0;


    while(isFrame)
    {
        const predictions = await model.estimateFaces({
            input: document.querySelector("#webStream")
        });
        if (predictions.length > 0)
        {

            let mesh = predictions[0].mesh;

            // console.log('predicted ' + mesh.length);
            // drawPoints(mesh);
            // drawAllPoints(mesh);

            ear = eye_feature(mesh);
            mar = mouth_feature(mesh);
            moe = mouth_over_eye_feature(mar, ear);

            ear = (ear - ears_norm[0])/ears_norm[1]
            mar = (mar - mars_norm[0])/mars_norm[1]
            moe = (moe - moes_norm[0])/moes_norm[1]

            if (ear_main == -1000)
            {
                ear_main = ear;
                mar_main = mar;
                moe_main = moe;
            }
            else
            {
                ear_main = ear_main*decay + ear*(1-decay);
                mar_main = mar_main*decay + mar*(1-decay);
                moe_main = moe_main*decay + moe*(1-decay);
            }
            
            console.log('ear ' + ear_main + ' mar ' + mar_main + ' moe ' + moe_main);

            if (moe_main > drowsy_threshold)
            {
                new_state = states2ind['drowsy'];
            }
            else if (moe_main < alert_threshold)
            {
                new_state = states2ind['alert'];
            }
            else
            {
                new_state = states2ind['neutral'];
            }

            if (new_state == old_state)
            {
                frame_count = frame_count + 1;
            }
            else
            {
                new_frame_count = new_frame_count + 1;
                frame_count = Math.max(frame_count - 1, 0);
                if (new_frame_count > 2)
                {
                    new_frame_count = 0;
                    frame_count = 0;
                    old_state = new_state;
                }
            }

            if (frame_count > state_frames_threshold)
            {
                console.log('state is ' + ind2states[new_state]);
                result_state = ind2states[new_state];
                state_para.innerHTML = 'State: ' + result_state;
                state_para.style.display = 'block';
            }
            else
            {
                state_para.style.display = 'none';
            }


            gaze_lr = eye_lr(mesh);
            console.log('gaze_lr' + gaze_lr);

            if (gaze_lr < gaze_left_threshold)
            {
                gaze_new_state = gazeStates2ind['left'];
            }
            else if (gaze_lr > gaze_right_threshold)
            {
                gaze_new_state = gazeStates2ind['right'];
            }
            else
            {
                gaze_new_state = gazeStates2ind['center'];
            }
            
            if (gaze_new_state == gaze_old_state)
            {
                gaze_frame_count = gaze_frame_count + 1;
            }
            else
            {
                gaze_new_frame_count = gaze_new_frame_count + 1;
                gaze_frame_count = Math.max(gaze_frame_count - 1, 0);
                if (gaze_new_frame_count > 2)
                {
                    gaze_new_frame_count = 0;
                    gaze_frame_count = 0;
                    gaze_old_state = gaze_new_state;
                }
            }

            if(gaze_new_state == gazeStates2ind['center'])
            {
                console.log('gaze is center');
                gaze_para.innerHTML = 'Gaze: Center';
                gaze_para.style.display = 'block';
            }
            else if (gaze_frame_count > gaze_frames_threshold)
            {
                console.log('gaze is ' + ind2gazeStates[gaze_new_state]);
                result_gaze = ind2gazeStates[gaze_new_state];
                gaze_para.innerHTML = 'Have been gazing: ' + result_gaze + ' . Please study';
                gaze_para.style.display = 'block';
            }
            else
            {
                gaze_para.style.display = 'none';
            }
        } 
    }
    if(!isFrame)
    {
        stopModel();
    }
}

function stopModel(){
    console.log('stopping Model');
    isFrame = false;
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.beginPath();
    ear_main = -1000;
    mar_main = -1000;
    moe_main = -1000;
    gaze_para.style.display = 'none';
    state_para.style.display = 'none';
}

function calibrate(){
    alert("Starting Calibration, Please look at the screen.");
    startCalib();
}

async function startCalib()
{
    if(!model)
    {
        console.log('loading model');
        model = await faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);
        console.log('model loaded');
    }
    console.log('starting calibration. Please stay in neutral position');
    timer.style.display = 'block';
    results_div.style.display = 'block';
    ears = [];
    mars = [];
    moes = [];
    timer.innerHTML = "Starting Calibration, Please look at the screen."
    while (ears.length < calib_frames)
    {
        timer.innerHTML = "Calibrating..." + ears.length + "/50";
        const predictions = await model.estimateFaces({
            input: document.querySelector("#webStream")
        });
        if (predictions.length > 0)
        {

            let mesh = predictions[0].mesh;
            drawAllPoints(mesh);
            ear = eye_feature(mesh);
            mar = mouth_feature(mesh);
            moe = mouth_over_eye_feature(mar, ear);
            console.log('ear ' + ear + ' mar ' + mar + ' moe ' + moe);
            ears.push(ear);
            mars.push(mar);
            moes.push(moe);
        } 
    }
    console.log('stopping calibration');
    ears_norm[0] = average(ears);
    ears_norm[1] = standardDeviation(ears);
    mars_norm[0] = average(mars);
    mars_norm[1] = standardDeviation(mars);
    moes_norm[0] = average(moes);
    moes_norm[1] = standardDeviation(moes);

    if (ears.length >= calib_frames)
    {
        stopModel();
    }

    console.log('calibration completed');
    timer.innerHTML = "Calibration Completed.";
    console.log(ears_norm);
    console.log(mars_norm);
    console.log(moes_norm);
    timer.style.display = 'none';
    results_div.style.display ='none';
}

function wakeUp() {
    if (!vidPlayer.paused) {
      vidPlayer.pause();
      alert("Wake up");
    }
}

function triggerModel() {
    var checkBox = document.getElementById("modelSwitch");
  
    if (checkBox.checked == true){
      startModel();
    } else {
      stopModel();
    }
}

document.addEventListener("DOMContentLoaded", function(event)
{ 
    startCamera();
    state_para = document.getElementById('state_para');
    gaze_para = document.getElementById('gaze_para');
    timer = document.getElementById("timer");
    vidPlayer = document.getElementById("player");
    results_div = document.getElementById("result_div");
});

function modelBasedUI() {
    var modelType = document.getElementById("modelType").value;
    myWrappers = [
        document.getElementById("jsModel"),
        document.getElementById("python")
     ];
    for (i=0; i<myWrappers.length; i++){    
        if(dropDown.value === "default"){ 
           myWrappers[i].style.display = "none";
         } else if(dropDown.value === "0" || dropDown.value === "1" || dropDown.value === "2"){
           myWrappers[i].style.display = "none";
           myWrappers[0].style.display = "block";
          }
     }
}
