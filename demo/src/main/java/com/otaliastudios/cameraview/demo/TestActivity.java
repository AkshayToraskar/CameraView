package com.otaliastudios.cameraview.demo;

import android.os.Bundle;

import com.otaliastudios.cameraview.CameraLogger;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.Frame;
import com.otaliastudios.cameraview.FrameProcessor;
import com.otaliastudios.cameraview.TensorflowCamActivity;
import com.otaliastudios.tfcustomview.OverlayView;

import androidx.annotation.NonNull;

public class TestActivity extends TensorflowCamActivity {

    CameraView cameraView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);
        CameraLogger.setLogLevel(CameraLogger.LEVEL_VERBOSE);
        cameraView = (CameraView) findViewById(R.id.camera);
        cameraView.setLifecycleOwner(this);

        cameraView.addFrameProcessor(new FrameProcessor() {
            @Override
            public void process(@NonNull Frame frame) {
                previewFrame(frame);
            }
        });
    }


    @Override
    public void setTFValues() {

        OverlayView trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);

        setValues(trackingOverlay,
                "detect_custom.tflite",
                "file:///android_asset/labels_custom.txt",
                300,
                0.3f);
    }


}
