package com.google.firebase.samples.apps.mlkit.common;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera.CameraInfo;
import androidx.annotation.Nullable;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.util.Log;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/** Utils functions for bitmap conversions. */
public class BitmapUtils {

    // Convert NV21 format byte buffer to bitmap.
    @Nullable
    public static Bitmap getBitmap(ByteBuffer data, FrameMetadata metadata) {
        data.rewind();
        byte[] imageInBuffer = new byte[data.limit()];
        data.get(imageInBuffer, 0, imageInBuffer.length);
        try {
            YuvImage image =
                    new YuvImage(
                            imageInBuffer, ImageFormat.NV21, metadata.getWidth(), metadata.getHeight(), null);
            if (image != null) {
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                image.compressToJpeg(new Rect(0, 0, metadata.getWidth(), metadata.getHeight()), 50, stream);

                Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());

                stream.close();
                return rotateBitmap(bmp, metadata.getRotation(), metadata.getCameraFacing());
            }
        } catch (Exception e) {
            Log.e("VisionProcessorBase", "Error: " + e.getMessage());
        }
        return null;
    }

    public static Bitmap getBitmap(RenderScript rs, ScriptIntrinsicYuvToRGB yuvToRGB, ByteBuffer data, FrameMetadata metadata) {
        data.rewind();
        byte[] imageInBuffer = new byte[data.limit()];
        data.get(imageInBuffer, 0, imageInBuffer.length);

        int yuvDatalength = metadata.getWidth() * metadata.getHeight() * 3 / 2;
        Allocation aIn = Allocation.createSized(rs, Element.U8(rs), yuvDatalength);

        Bitmap bmpout = Bitmap.createBitmap(metadata.getWidth(), metadata.getHeight(), Bitmap.Config.ARGB_8888);
        Allocation aOut = Allocation.createFromBitmap(rs, bmpout);
        yuvToRGB.setInput(aIn);

        aIn.copyFrom(data);

        yuvToRGB.forEach(aOut);

        aOut.copyTo(bmpout);


        return bmpout;
    }

    // Rotates a bitmap if it is converted from a bytebuffer.
    private static Bitmap rotateBitmap(Bitmap bitmap, int rotation, int facing) {
        Matrix matrix = new Matrix();
        int rotationDegree = 0;
        switch (rotation) {
            case FirebaseVisionImageMetadata.ROTATION_90:
                rotationDegree = 90;
                break;
            case FirebaseVisionImageMetadata.ROTATION_180:
                rotationDegree = 180;
                break;
            case FirebaseVisionImageMetadata.ROTATION_270:
                rotationDegree = 270;
                break;
            default:
                break;
        }

        // Rotate the image back to straight.}
        matrix.postRotate(rotationDegree);
        if (facing == CameraInfo.CAMERA_FACING_BACK) {
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } else {
            // Mirror the image along X axis for front-facing camera image.
            matrix.postScale(-1.0f, 1.0f);
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }
    }
}

