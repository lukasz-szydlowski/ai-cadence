import Toybox.Graphics;
import Toybox.WatchUi;
import Toybox.System;
import Toybox.Time;
import Toybox.Time.Gregorian;
import Toybox.Application.Storage;

class UtilityAppSendView extends WatchUi.View {

    var _textArea = null;
    var _index = null;
    var _size = null;

    function initialize() {
        View.initialize();
    }

    // Load your resources here
    function onLayout(dc as Dc) as Void {
        _textArea = new WatchUi.TextArea({
            :text=> "",
            :color=>Graphics.COLOR_BLACK,
            :font=>[Graphics.FONT_MEDIUM],
            :locX =>WatchUi.LAYOUT_HALIGN_CENTER,
            :locY=>WatchUi.LAYOUT_VALIGN_CENTER,
            :width=>dc.getWidth() * 2/3,
            :height=>dc.getHeight() * 2/3
        });
    }

    // Called when this View is brought to the foreground. Restore
    // the state of this View and prepare it to be shown. This includes
    // loading resources into memory.
    function onShow() as Void 
    {
        var keys = Storage.getValue("keys");
        _index = -1;
        _size = keys.size();
        WatchUi.requestUpdate();

        send();
    }

    function send()
    {
        var keys = Storage.getValue("keys");

        if (keys == null)
        {
            return;
        }

        //var url = "https://api.keyval.aclapps.com/";
        var url = "https://kvdb.io/V6Apyn8NgYB9fE2p1eH6AA/";
        var options = 
        {
            :method => Communications.HTTP_REQUEST_METHOD_PUT,
            :headers => 
            {
                "Content-Type" => Communications.REQUEST_CONTENT_TYPE_JSON,
                //"Authorization" => "0acb6aee-f24b-4da8-a719-96b92de8f1a4",
            },
            :responseType => Communications.HTTP_RESPONSE_CONTENT_TYPE_JSON,
        };

        var responseCallback = method(:onReceive); 

        if (_index == -1)
        {
            Communications.makeWebRequest(url + "keys", { "keys" => keys }, options, responseCallback);
            return;
        }

        if (_index >= keys.size())
        {
            WatchUi.popView(WatchUi.SLIDE_IMMEDIATE);
            return;
        }

        var key = keys[_index];
        var data = Storage.getValue(key);
        Communications.makeWebRequest(url + key, data, options, responseCallback);
    }

    function onReceive(responseCode, responseData)
    {
        _index++;
        WatchUi.requestUpdate();

        send();
    }    

    // Update the view
    function onUpdate(dc as Dc) as Void {
        // Call the parent onUpdate function to redraw the layout
        dc.setColor(Graphics.COLOR_BLACK, Graphics.COLOR_WHITE);
        dc.clear();

        var percent = (100.0 * (_index + 1)) / (_size + 1);

        var text = "";
        text = text + "Sending: " + percent.format("%01d") + "%";
        text = text + "\n";

        _textArea.setText(text);
        _textArea.draw(dc);          
    }    
}