import Toybox.Graphics;
import Toybox.WatchUi;
import Toybox.System;
import Toybox.Time;
import Toybox.Time.Gregorian;
import Toybox.Application.Storage;

class UtilityAppView extends WatchUi.View {

    var _textArea = null;

    function initialize() {
        View.initialize();
    }

    // Load your resources here
    function onLayout(dc as Dc) as Void {
        //setLayout(Rez.Layouts.MainLayout(dc));

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
    function onShow() as Void {
    }

    // Update the view
    function onUpdate(dc as Dc) as Void {
        // Call the parent onUpdate function to redraw the layout
        dc.setColor(Graphics.COLOR_BLACK, Graphics.COLOR_WHITE);
        dc.clear();

        var text = "";

        var app = getApp();

        var count = 0;
        var keys = Storage.getValue("keys");

        if (keys != null)
        {
            count = keys.size();
        }

        if (app._isRecording == false)
        {
            text = text + "AI Cadence";
            text = text + "\n";


            text = text +"Packs: " + count + "\n";

            _textArea.setText(text);
            _textArea.draw(dc);       

            return; 
        }

        var time = Gregorian.info(Time.now(), Time.FORMAT_SHORT);
        text = "Time: " + time.hour.format("%02d") + ":" + time.min.format("%02d") + ":" + time.sec.format("%02d") + "\n";
        text = text + "Cadence: " + app._cadence + "\n";

        var x = 0;
        var y = 0;
        var z = 0;

        if (app._X != null)
        {
            for (var i = 0; i < app._X.size(); i++)
            {
                x = x + app._X[i];
                y = y + app._Y[i];
                z = z + app._Z[i];
            }

            x = x / app._X.size();
            y = y / app._X.size();
            z = z / app._X.size();
        }

        text = text +"X: " + x + "\n";
        text = text +"Y: " + y + "\n";
        text = text +"Z: " + z + "\n";
        text = text +"Sequences: " + app._index + "\n";
        text = text +"Packs: " + count + "\n";

        _textArea.setText(text);
        _textArea.draw(dc);        
    }

    // Called when this View is removed from the screen. Save the
    // state of this View here. This includes freeing resources from
    // memory.
    function onHide() as Void {
    }

}
