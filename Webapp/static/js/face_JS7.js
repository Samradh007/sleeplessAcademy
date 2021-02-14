let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let camera = null;

let model = null;
let model_loaded = null;
let model_type = null;

let isCalibFrame = null;
let calibFrameCount = 0;
let calibFramesThreshold = 25;
let ears_norm = [0, 1];
let mars_norm = [0, 1];
let pucs_norm = [0, 1];
let moes_norm = [0, 1];
let ears = [];
let mars = [];
let pucs = [];
let moes = [];

let frameBeforeRun = 0;
let decay = 0.9;
let ear_main = -1000;
let mar_main = -1000;
let puc_main = -1000;
let moe_main = -1000;
let input_data = [];

let imgWidth = 1280;
let imgHeight = 720;

let sleepyPoints = [];

let keypoints_dict = {
  upper_lip: [61, 39, 0, 269],
  lower_lip: [291, 181, 17, 405],
  upper_righteye: [33, 160, 159, 158],
  lower_righteye: [133, 144, 145, 153],
  upper_lefteye: [263, 387, 386, 385],
  lower_lefteye: [362, 373, 374, 380],
  iris_righteye: [473, 474, 475, 476, 477],
  iris_lefteye: [468, 469, 470, 471, 472],
  // right_iris_0 ==> center, iris_1 ==> right_end, iris_2 ==> top, iris_3 ==> left_end, iris_4 ==> bottom
  // left_iris_0 ==> center, iris_3 ==> right_end, iris_2 ==> top, iris_1 ==> left_end, iris_4 ==> bottom
};

function distance(p1, p2) {
  return (
    ((p1.x * imgWidth - p2.x * imgWidth) ** 2 +
      (p1.y * imgHeight - p2.y * imgHeight) ** 2) **
    0.5
  );
}

function eye_aspect_ratio(mesh, upper_eye_points, lower_eye_points) {
  N1 = distance(mesh[upper_eye_points[1]], mesh[lower_eye_points[1]]);
  N2 = distance(mesh[upper_eye_points[2]], mesh[lower_eye_points[2]]);
  N3 = distance(mesh[upper_eye_points[3]], mesh[lower_eye_points[3]]);
  D = distance(mesh[upper_eye_points[0]], mesh[lower_eye_points[0]]);
  return (N1 + N2 + N3) / (3 * D);
}

function eye_feature(mesh) {
  return (
    (eye_aspect_ratio(
      mesh,
      keypoints_dict["upper_righteye"],
      keypoints_dict["lower_righteye"]
    ) +
      eye_aspect_ratio(
        mesh,
        keypoints_dict["upper_lefteye"],
        keypoints_dict["lower_lefteye"]
      )) /
    2
  );
}

function mouth_feature(mesh) {
  N1 = distance(
    mesh[keypoints_dict["upper_lip"][1]],
    mesh[keypoints_dict["lower_lip"][1]]
  );
  N2 = distance(
    mesh[keypoints_dict["upper_lip"][2]],
    mesh[keypoints_dict["lower_lip"][2]]
  );
  N3 = distance(
    mesh[keypoints_dict["upper_lip"][3]],
    mesh[keypoints_dict["lower_lip"][3]]
  );
  D = distance(
    mesh[keypoints_dict["upper_lip"][0]],
    mesh[keypoints_dict["lower_lip"][0]]
  );
  return (N1 + N2 + N3) / (3 * D);
}

function pupil_circularity(mesh, upper_eye_points, lower_eye_points) {
  perimeter =
    distance(mesh[upper_eye_points[0]], mesh[upper_eye_points[1]]) +
    distance(mesh[upper_eye_points[1]], mesh[upper_eye_points[2]]) +
    distance(mesh[upper_eye_points[2]], mesh[upper_eye_points[3]]) +
    distance(mesh[upper_eye_points[3]], mesh[lower_eye_points[0]]) +
    distance(mesh[lower_eye_points[0]], mesh[lower_eye_points[1]]) +
    distance(mesh[lower_eye_points[1]], mesh[lower_eye_points[2]]) +
    distance(mesh[lower_eye_points[2]], mesh[lower_eye_points[3]]) +
    distance(mesh[lower_eye_points[3]], mesh[upper_eye_points[0]]);
  area =
    Math.PI *
    (distance(mesh[upper_eye_points[1]], mesh[lower_eye_points[3]]) * 0.5) ** 2;
  return (4 * Math.PI * area) / perimeter ** 2;
}

function pupil_feature(mesh) {
  return (
    (pupil_circularity(
      mesh,
      keypoints_dict["upper_righteye"],
      keypoints_dict["lower_righteye"]
    ) +
      pupil_circularity(
        mesh,
        keypoints_dict["upper_lefteye"],
        keypoints_dict["lower_lefteye"]
      )) /
    2
  );
}

function mouth_over_eye_feature(mouth_aspect_ratio, eye_aspect_ratio) {
  return mouth_aspect_ratio / eye_aspect_ratio;
}

function average(data) {
  var sum = data.reduce(function (sum, value) {
    return sum + value;
  }, 0);

  var avg = sum / data.length;
  return avg;
}

function standardDeviation(values) {
  var avg = average(values);

  var squareDiffs = values.map(function (value) {
    var diff = value - avg;
    var sqrDiff = diff * diff;
    return sqrDiff;
  });

  var avgSquareDiff = average(squareDiffs);

  var stdDev = Math.sqrt(avgSquareDiff);
  return stdDev;
}

function onResults(results) {
  let ear = -1000;
  let mar = -1000;
  let puc = -1000;
  let moe = -1000;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );
  if (results.multiFaceLandmarks) {
    landmarks = results.multiFaceLandmarks[0];

    drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {
      color: "#C0C0C070",
      lineWidth: 1,
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {
      color: "#E0E0E0",
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {
      color: "#E0E0E0",
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {
      color: "#E0E0E0",
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {
      color: "#E0E0E0",
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {
      color: "#E0E0E0",
    });
    drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, { color: "#E0E0E0" });

    ear = eye_feature(landmarks);
    mar = mouth_feature(landmarks);
    puc = pupil_feature(landmarks);
    moe = mouth_over_eye_feature(mar, ear);
  }

  if (isCalibFrame) {
    calibrate(ear, mar, puc, moe);
  } else {
    inference(ear, mar, puc, moe);
  }

  canvasCtx.restore();
}

function calibrate(ear, mar, puc, moe) {
  if (calibFrameCount < calibFramesThreshold) {
    ears.push(ear);
    mars.push(mar);
    pucs.push(puc);
    moes.push(moe);
  } else {
    ears_norm[0] = average(ears);
    ears_norm[1] = standardDeviation(ears);
    mars_norm[0] = average(mars);
    mars_norm[1] = standardDeviation(mars);
    pucs_norm[0] = average(pucs);
    pucs_norm[1] = standardDeviation(pucs);
    moes_norm[0] = average(moes);
    moes_norm[1] = standardDeviation(moes);

    // #### Show to user
    console.log("Completed Calibration");
    timer.innerHTML = "Calibration Completed.";
    timer.style.display = "none";
    results_div.style.display = "none";
    isCalibFrame = false;
    stopModel();
    ears = [];
    mars = [];
    pucs = [];
    moes = [];
  }
  timer.innerHTML =
    "Calibrating..." + calibFrameCount + "/" + calibFramesThreshold;
  calibFrameCount = calibFrameCount + 1;
}

function inference(ear, mar, puc, moe) {
  if (ear != -1000) {
    ear = (ear - ears_norm[0]) / ears_norm[1];
    mar = (mar - mars_norm[0]) / mars_norm[1];
    puc = (puc - pucs_norm[0]) / pucs_norm[1];
    moe = (moe - moes_norm[0]) / moes_norm[1];
    if (ear_main == -1000) {
      ear_main = ear;
      mar_main = mar;
      puc_main = puc;
      moe_main = moe;
    } else {
      ear_main = ear_main * decay + (1 - decay) * ear;
      mar_main = mar_main * decay + (1 - decay) * mar;
      puc_main = puc_main * decay + (1 - decay) * puc;
      moe_main = moe_main * decay + (1 - decay) * moe;
    }
  } else {
    ear_main = -1000;
    mar_main = -1000;
    puc_main = -1000;
    moe_main = -1000;
  }

  if (input_data.length == 20) {
    input_data.shift();
  }
  input_data.push([ear_main, mar_main, puc_main, moe_main]);

  frameBeforeRun = frameBeforeRun + 1;

  if (frameBeforeRun >= 15 && input_data.length == 20) {
    frameBeforeRun = 0;
    // #### #### Call label method
    getOurLabel();
  }
}

async function startCalib() {
  isCalibFrame = true;
  calibFrameCount = 0;
  ears_norm = [0, 1];
  mars_norm = [0, 1];
  pucs_norm = [0, 1];
  moes_norm = [0, 1];
  ears = [];
  mars = [];
  pucs = [];
  moes = [];

  // Await now working exactly
  if (!model_loaded) {
    console.log("loading model");
    model_loaded = await new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });
    model_loaded.setOptions({
      maxNumFaces: 1,
      minDetectionConfidence: 0.3,
      minTrackingConfidence: 0.8,
    });
    console.log("model loaded");
    model_loaded.onResults(onResults);
  }
  model = model_loaded;

  // #### Show to user
  console.log("starting calibration. Please stay in neutral position");
  timer.innerHTML = "Starting Calibration, Please look at the screen.";
}

function stopModel() {
  model = null;
  console.log("model stoppped");
}

function startModel() {
  isCalibFrame = false;
  frameBeforeRun = 0;
  ear_main = -1000;
  mar_main = -1000;
  puc_main = -1000;
  moe_main = -1000;
  input_data = [];
  model = model_loaded;
  console.log("model started");
}

function wakeUp() {
  if (!vidPlayer.paused) {
    vidPlayer.pause();
    alert("Wake up");
  }
}

function storeLabel(labels) {
  if(labels == 1 && !vidPlayer.paused){
    sleepyPoints.push(vidPlayer.currentTime);
  }
  if(sleepyPoints.length > 10){
    sleepyPoints = [];
  }
  console.log(sleepyPoints);
}

function triggerModel() {
  var checkBox = document.getElementById("modelSwitch");

  if (calibFrameCount != 0) {
    if (checkBox.checked == true) {
      startModel();
    } else {
      stopModel();
    }
  } else {
    alert("Please calibrate before using the alertness detection.");
    checkBox.checked = false;
  }
}

function getOurLabel(){

  var send = {
      input_data: input_data,
      modelType: model_type.value,
      };
    
    $.getJSON("/get_label", send, function (response) {
      console.log(response.label);
      storeLabel(response.label);
      if(response.label ==1){
        wakeUp();
      }
    });
      
};

document.addEventListener("DOMContentLoaded", function (event) {
  videoElement = document.getElementsByClassName("webStream")[0];
  canvasElement = document.getElementsByClassName("outputCanvas")[0];
  canvasCtx = canvasElement.getContext("2d");
  timer = document.getElementById("timer");
  vidPlayer = document.getElementById("player");
  model_type = document.getElementById("modelType");
  results_div = document.getElementById("result_div");
  imgHeight = canvasElement.height;
  imgWidth = canvasElement.width;
  // data to send back to the server

  camera = new Camera(videoElement, {
    onFrame: async () => {
      if (model) {
        await model.send({ image: videoElement });
      } else {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(
          videoElement,
          0,
          0,
          canvasElement.width,
          canvasElement.height
        );
      }
    },
    width: imgWidth,
    height: imgHeight,
  });
  camera.start();
});
