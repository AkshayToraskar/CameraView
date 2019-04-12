package com.otaliastudios.cameraview;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.widget.Toast;

import com.otaliastudios.tfcustomview.OverlayView;
import com.otaliastudios.tf.env.BorderedText;
import com.otaliastudios.tf.env.ImageUtils;
import com.otaliastudios.tf.env.Logger;
import com.otaliastudios.tf.tflite.Classifier;
import com.otaliastudios.tf.tflite.TFLiteObjectDetectionAPIModel;
import com.otaliastudios.tf.tracking.MultiBoxTracker;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

public abstract class TensorflowCamActivity extends AppCompatActivity {


    private static final Logger LOGGER = new Logger();
    // Configuration values for the prepackaged SSD model.
    private static int TF_OD_API_INPUT_SIZE;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static String TF_OD_API_MODEL_FILE ;
    private static String TF_OD_API_LABELS_FILE ;
    private static final TensorflowCamActivity.DetectorMode MODE = TensorflowCamActivity.DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Classifier detector;
    private Bitmap croppedBitmap = null;
    private boolean isProcessingFrame = false;
    private int[] rgbBytes = null;
    private byte[][] yuvBytes = new byte[3][];
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private int yRowStride;
    protected int previewWidth;
    protected int previewHeight;
    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private boolean computingDetection = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private byte[] luminanceCopy;
    private BorderedText borderedText;
    private OverlayView trackingOverlay;
    private boolean debug = false;
    private Handler handler;
    private HandlerThread handlerThread;
    private Integer sensorOrientation;

    public void previewFrame(final Frame frame) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                //Size previewSize = cameraView.getPictureSize();
                previewHeight = frame.getSize().getHeight();
                previewWidth = frame.getSize().getWidth();
                rgbBytes = new int[previewWidth * previewHeight];

                LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
                rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
                onPreviewSizeChosen(new Size(frame.getSize().getWidth(), frame.getSize().getHeight()), 90);

            }
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = frame.getData();
        yRowStride = previewWidth;

        imageConverter =
                new Runnable() {
                    @Override
                    public void run() {
                        ImageUtils.convertYUV420SPToARGB8888(frame.getData(), previewWidth, previewHeight, rgbBytes);
                    }
                };

        postInferenceCallback =
                new Runnable() {
                    @Override
                    public void run() {
                        //camera.addCallbackBuffer(bytes);
                        isProcessingFrame = false;
                    }
                };
        processImage();
    }


    public void processImage() {

        //Log.v("TIMING", "Bitmap generation started " + getTimeStamp());
        //final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
        //Log.v("TIMING", "Bitmap generation Ended " + getTimeStamp());


        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();
                        //boolean unknown = false, cervix = false, os = false;

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {

                                //if (result.getTitle().equals("unknown") && !unknown) {
                                //  unknown = true;
                                canvas.drawRect(location, paint);
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);

                                /*} else if (result.getTitle().equals("Cervix") && !cervix) {
                                    canvas.drawRect(location, paint);
                                    cropToFrameTransform.mapRect(location);
                                    result.setLocation(location);
                                    mappedRecognitions.add(result);
                                    cervix = true;

                                } else if (result.getTitle().equals("Os") && !os) {
                                    canvas.drawRect(location, paint);
                                    cropToFrameTransform.mapRect(location);
                                    result.setLocation(location);
                                    mappedRecognitions.add(result);
                                    os = true;
                                }*/


                            }
                        }

                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();


                        computingDetection = false;

                    }
                });


    }


    public void onPreviewSizeChosen(Size size, int rotation) {

        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        //trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        //trackingOverlay = getTrackingOverlay();
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

    }


    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }


    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }


    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }


    public boolean isDebug() {
        return debug;
    }


    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    protected int getLuminanceStride() {
        return yRowStride;
    }

    protected byte[] getLuminance() {
        return yuvBytes[0];
    }


    private enum DetectorMode {
        TF_OD_API;
    }

    public String getTimeStamp() {
        //Date d = new Date();
        return new SimpleDateFormat("dd.MM.yyyy HH.mm.ss").format(new Date());
    }


    @Override
    protected void onStart() {
        super.onStart();

        setTFValues();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }


    public abstract void setTFValues();


    //gettter setter
    public void setValues(OverlayView trackingOverlay, String modelFile, String lableFile, int size, float minConfidance) {
        this.trackingOverlay = trackingOverlay;
        TF_OD_API_MODEL_FILE = modelFile;
        TF_OD_API_LABELS_FILE = lableFile;
        TF_OD_API_INPUT_SIZE = size;
        MINIMUM_CONFIDENCE_TF_OD_API=minConfidance;
    }


}
