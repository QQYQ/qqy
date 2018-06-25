/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package gdut.bsx.tensorflowtraining.ternsorflow;

import android.graphics.Bitmap;
import android.graphics.RectF;
import java.util.List;

/**
 *与不同识别引擎交互的通用接口。
 */
public interface Classifier {
    /**
     描述识别内容的分类器返回的不可变结果。
     */
    public class Recognition {
        /**
         *识别的唯一标识符。特定于类，而不是类的实例
         *对象。
         */
        private final String id;

        /**
         * 显示识别名称。
         */
        private final String title;

        /**
         * 一种可分类的分数，用来表示识别相对于他人的好坏程度。更高的应该更好。
         */
        private final Float confidence;

        /** 在源映像中为识别对象的位置可选的位置。*/
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    List<Recognition> recognizeImage(Bitmap bitmap);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();
}