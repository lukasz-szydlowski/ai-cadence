import Toybox.Application;
import Toybox.Lang;
import Toybox.WatchUi;
import Toybox.Graphics;
import Toybox.System;
import Toybox.Sensor;
import Toybox.Time;
import Toybox.Time.Gregorian;
import Toybox.Communications;
import Toybox.Application.Storage;

class UtilityApp extends Application.AppBase {

    var _isRecording = false;
    var _index = 0;
    var _cadence = 0;
    var _X = null;
    var _Y = null;
    var _Z = null;
    var _data = {};
    var _packs = 0;

    function initialize() {
        AppBase.initialize();
    }

    function onStart(state as Dictionary?) as Void {
    }

    function onStop(state as Dictionary?) as Void {
    }

    function start()
    {
        Sensor.enableSensorEvents(method( :onCadence ));

        var maxSampleRate = Sensor.getMaxSampleRate();
        var options = {:period => 1, :sampleRate => maxSampleRate, :enableAccelerometer => true};
        Sensor.registerSensorDataListener(method(:onAccelerometer), options);

        _isRecording = true;
        WatchUi.requestUpdate();
    }

    function stop()
    {
        Sensor.enableSensorEvents(null);
        Sensor.unregisterSensorDataListener();

        _isRecording = false;
        WatchUi.requestUpdate();
    }

    function send()
    {
        WatchUi.switchToView(new UtilityAppSendView(), null, WatchUi.SLIDE_IMMEDIATE);
    }

    function clear()
    {
        Storage.clearValues();
    }

    function onCadence(sensorInfo as Sensor.Info) as Void {
        _cadence = sensorInfo.cadence;

        if (_cadence == null)
        {
            _cadence = 0;
        }

        WatchUi.requestUpdate();
    }

    function onAccelerometer(sensorData as SensorData) as Void {
        _index++;

        _X = sensorData.accelerometerData.x;
        _Y = sensorData.accelerometerData.y;
        _Z = sensorData.accelerometerData.z;

        WatchUi.requestUpdate();

        var item = 
        {
            "Cadence" => _cadence,
            "X" => _X,
            "Y" => _Y,
            "Z" => _Z
        };

        _data.put(_index, item);

        if (_index % 60 != 0) {
            return;
        }

        var time = Gregorian.info(Time.now(), Time.FORMAT_SHORT);
        var key = Lang.format("$1$_$2$_$3$_$4$_$5$_$6$_$7$", 
            [time.year.format("%04d"), time.month.format("%02d"), time.day.format("%02d"), time.hour.format("%02d"), time.min.format("%02d"), time.sec.format("%02d"), _index.format("%05d")]);

        var keys = Storage.getValue("keys");
        
        if (keys == null)
        {
            keys = new[1];
            keys[0] = key;

            Storage.setValue("keys", keys);
        }
        else
        {
            var size = keys.size();
            var newKeys = new [size + 1];

            for (var i = 0; i < size; i++)
            {
                newKeys[i] = keys[i];
            }

            newKeys[size] = key;
            Storage.setValue("keys", newKeys);
        }

        Storage.setValue(key, _data);

        _data = {};
        _packs++;

        WatchUi.requestUpdate();
    }

    function getInitialView() as Array<Views or InputDelegates>? {
         return [ new UtilityAppView(), new UtilityAppDelegate()] as Array<Views or InputDelegates>;
    }
}

function getApp() as UtilityApp 
{
    return Application.getApp() as UtilityApp;
}